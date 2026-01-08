import torch
import triton
import triton.language as tl
import time

# ==============================================================================
# Triton Kernel for Fused HSTU Attention with KV Cache Support
# ==============================================================================

@triton.autotune(
     configs=[
         triton.Config({'BLOCK_M':32,'BLOCK_N':32},num_warps=4,num_stages=2),
         triton.Config({'BLOCK_M':32, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M':32,'BLOCK_N':64},num_warps=4,num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
      ],
      key=['N_q', 'N_kv', 'HEAD_DIM'], # Key parameters that affect performance
  )
@triton.jit
def _hstu_fwd_kernel(
    Q, K, V, U, MASK, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_uz, stride_uh, stride_um, stride_uk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_maskz, stride_maskm, stride_maskn, # Mask is (B, N_q, N_kv), no head dim
    Z, H, N_q, N_kv,
    avg_factor,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton Kernel for HSTU Attention.
    Computes O = ((silu(Q @ K.T / avg_factor) * MASK) @ V) * U
    """
    # 1. Program and Offset Setup
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Q, K, V, U, O pointers for the current head and batch
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    u_offset = off_z * stride_uz + off_h * stride_uh
    o_offset = off_z * stride_oz + off_h * stride_oh
    # Mask pointer for the current batch item (broadcast across heads)
    mask_offset = off_z * stride_maskz

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_kv),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_kv, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    U_block_ptr = tl.make_block_ptr(
        base=U + u_offset,
        shape=(N_q, HEAD_DIM),
        strides=(stride_um, stride_uk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_q, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    Mask_block_ptr = tl.make_block_ptr(
        base=MASK + mask_offset,
        shape=(N_q, N_kv),
        strides=(stride_maskm, stride_maskn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # 2. Initialize accumulator for the output tile
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 3. Load Q tile (it's constant for the inner loop)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # 4. Loop over K and V sequence length
    for start_n in range(0, N_kv, BLOCK_N):
        # 4a. Load K and V tiles
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        
        # 4b. Compute Q @ K.T tile
        s = tl.dot(q, k)
        
        # 4c. Apply SiLU and scaling
        s = s / avg_factor
        s = s*tl.sigmoid(s)

        # 4d. Apply mask
        mask = tl.load(Mask_block_ptr, boundary_check=(0, 1))
        s = s * mask
        
        # 4e. Compute S @ V and update accumulator
        acc += tl.dot(s.to(v.dtype), v)

        # 4f. Update K, V, Mask pointers for the next iteration
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (0, BLOCK_N))

    # 5. Load U, multiply with accumulator, and store output
    u = tl.load(U_block_ptr, boundary_check=(0, 1))
    acc = acc * u
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty), boundary_check=(0, 1))

# ==============================================================================
# Python Wrapper for the Triton Kernel (FOR WARM-UP ONLY)
# ==============================================================================

def hstu_attention_triton_warmup(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, u: torch.Tensor, mask: torch.Tensor, avg_factor: float):
    """
    A simple Python wrapper for HSTU attention, used ONLY for warming up the
    Triton autotuner. It should not be called inside a torch.compile region.
    """
    B, N_q, H, D = q.shape
    _, N_kv, _, _ = k.shape

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    u = u.transpose(1, 2).contiguous()
    mask = mask.contiguous()
    o = torch.empty_like(q)

    def grid(meta):
        return (triton.cdiv(N_q, meta['BLOCK_M']), B * H)

    _hstu_fwd_kernel[grid](
        q, k, v, u, mask, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        u.stride(0), u.stride(1), u.stride(2), u.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2),
        B, H, N_q, N_kv,
        avg_factor,
        HEAD_DIM=D,
        BLOCK_DMODEL=D,
    )
    return o.transpose(1, 2)


# ==============================================================================
# Helper Function to Print Autotune Results
# ==============================================================================
def print_autotune_results(kernel):
    """Prints the best configurations found by the autotuner."""
    print("--- Autotune Best Configurations ---")
    if not hasattr(kernel, 'cache') or not kernel.cache:
        print("Autotuner cache is empty. Make sure the kernel has been run (warmed up).")
        return

    for key, config in kernel.cache.items():
        # FIX: Correctly unpack the key tuple
        n_q, n_kv, head_dim = key[0:3]
        print(f"For input shape (N_q={n_q}, N_kv={n_kv}, HEAD_DIM={head_dim}):")
        print(f"  -> Best config: {config}")
    print("-" * 34 + "\n")

def calculate_flops(B, H, D, L, user_length, target_length, micorbatch_size):
    """
     HSTU Attention  FLOPs 
    """
    def flops_op(N_q, N_kv, B, H, D):
        """hstu_attention  FLOPs。"""
        matmul_flops = 4 * B * H * N_q * N_kv * D
        elementwise_flops_sv = 7 * B * H * N_q * N_kv
        elementwise_flops_u = B * H * N_q * D
        return matmul_flops + elementwise_flops_sv + elementwise_flops_u

    # No KV Cache: one big operation
    flops_no_kv = flops_op(L, L, B, H, D)

    # Sliding Window KV Cache
    cache_size = user_length + micorbatch_size
    # Initial fill
    flops_prefill = flops_op(cache_size, cache_size, B, H, D)
    
    # Decode loop
    flops_decode_total = 0
    for _ in range(cache_size, L, micorbatch_size):
        # Each decode step attends to the full cache
        flops_decode_total += flops_op(micorbatch_size, cache_size, B, H, D)

    flops_with_kv = flops_prefill + flops_decode_total
    return flops_no_kv, flops_with_kv

# ==============================================================================
# Test Function to Benchmark Performance
# ==============================================================================

def test_hstu_attention_long_target_triton(num_repeat: int = 100,print_flops: bool = True):
    # --- Test Configuration ---
    B, H, D = 1, 3, 256
    user_length = 3171
    target_length = 4096
    micorbatch_size = 1024
    L = user_length + target_length
    
    print(f"--- Configuration ---")
    print(f"Batch={B}, Heads={H}, Dim={D}")
    print(f"User Length={user_length}, Target Length={target_length}, Total Length={L}")
    print(f"Microbatch Size={micorbatch_size}\n")

    dtype = torch.float16
    q = torch.randn(B, L, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, L, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, L, H, D, dtype=dtype, device="cuda")
    u = torch.randn(B, L, H, D, dtype=dtype, device="cuda")
    mask = torch.zeros((B, L, L), dtype=dtype, device="cuda")
    mask[:,:,:user_length]=1

    # --- WARM-UP PHASE ---
    print("--- Warming up Triton autotuner ---")
    # Run with representative shapes to populate the autotuner cache
    _ = hstu_attention_triton_warmup(q, k, v, u, mask, avg_factor=L)
    cache_size = user_length + micorbatch_size
    if cache_size <= L:
        _ = hstu_attention_triton_warmup(q[:, :cache_size], k[:, :cache_size], v[:, :cache_size], u[:, :cache_size], mask[:, :cache_size, :cache_size], avg_factor=cache_size)
    print("Warm-up complete. Starting benchmarks.\n")
    print_autotune_results(_hstu_fwd_kernel)

    # FIX: Robustly extract the first available configuration from the cache.
    # This avoids the "Could not find a cached config" warning and ensures a complete config.
    if _hstu_fwd_kernel.cache:
        best_config = next(iter(_hstu_fwd_kernel.cache.values()))
    else:
        print("FATAL: Autotuner cache is empty after warm-up. Using a default config.")
        best_config = _hstu_fwd_kernel.configs[0]
    
    # Reconstruct the full config dictionary from the Triton Config object
    config_kwargs = best_config.kwargs.copy()
    config_kwargs['num_warps'] = best_config.num_warps
    config_kwargs['num_stages'] = best_config.num_stages
    
    print(f"Using fixed config for compiled functions: {config_kwargs}\n")
    
    flops_no_kv, flops_with_kv = calculate_flops(B, H, D, L, user_length, target_length, micorbatch_size)

    # --- 1. Benchmark: No KV Cache (Compiled) ---
    # For this static case, we can use the simpler `[grid]` syntax, which is more
    # idiomatic for torch.compile and avoids potential conflicts with .run().
    @torch.compile(fullgraph=True)
    def compiled_no_kv(q, k, v, u, mask, avg_factor):
        B, N_q, H, D = q.shape
        _, N_kv, _, _ = k.shape
        
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        u_t = u.transpose(1, 2).contiguous()
        o_t = torch.empty_like(q_t)
        
        grid = lambda meta: (triton.cdiv(N_q, meta['BLOCK_M']), B * H)
        
        _hstu_fwd_kernel[grid](
            q_t, k_t, v_t, u_t, mask, o_t,
            q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),
            k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
            v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),
            u_t.stride(0), u_t.stride(1), u_t.stride(2), u_t.stride(3),
            o_t.stride(0), o_t.stride(1), o_t.stride(2), o_t.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2),
            B, H, N_q, N_kv, avg_factor, HEAD_DIM=D, BLOCK_DMODEL=D
        )
        return o_t.transpose(1, 2)

    print("--- Compiling 'No KV Cache' function ---")
    _ = compiled_no_kv(q, k, v, u, mask, float(L)) # Compile
    torch.cuda.synchronize()
    print("Compilation complete.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(num_repeat):
        _ = compiled_no_kv(q, k, v, u, mask, float(L))
    torch.cuda.synchronize()
    end_time = time.time()
    
    time_no_kv = (end_time - start_time) / num_repeat
    peak_mem_no_kv = torch.cuda.max_memory_allocated() / 1e6
    tflops_no_kv = flops_no_kv / time_no_kv / 1e12

    print("\n--- Analysis: Without KV Cache (Compiled) ---")
    print(f"  Avg. Time:            {time_no_kv:.6f} s")
    print(f"  Peak Memory:          {peak_mem_no_kv:.2f} MB")
    print(f"  Theoretical FLOPs:    {flops_no_kv / 1e12:.4f} TFLOPs")
    print(f"  Achieved Performance: {tflops_no_kv:.4f} TFLOP/s")

    # --- 2. Benchmark: With Sliding Window KV Cache (Compiled) ---
    # For the dynamic case, we stick with the .run() method and a fixed config
    # to ensure a static graph for torch.compile.
    @torch.compile(fullgraph=True)
    def compiled_sliding_window_kv(q, k, v, u, mask, k_cache, v_cache, cache_sz, micro_bs, config):
        # Pre-allocate full output tensor to avoid torch.cat
        output = torch.empty_like(q)

        # --- Initial Fill Phase ---
        q_fill = q[:, :cache_sz, :, :]
        k_fill = k[:, :cache_sz, :, :]
        v_fill = v[:, :cache_sz, :, :]
        u_fill = u[:, :cache_sz, :, :]
        mask_fill = mask[:, :cache_sz, :cache_sz]
        
        B, N_q_fill, H, D = q_fill.shape
        
        q_f = q_fill.transpose(1, 2).contiguous()
        k_f = k_fill.transpose(1, 2).contiguous()
        v_f = v_fill.transpose(1, 2).contiguous()
        u_f = u_fill.transpose(1, 2).contiguous()
        o_f = torch.empty_like(q_f)
        
        grid_fill = lambda meta: (triton.cdiv(N_q_fill, meta['BLOCK_M']), B * H)
        
        _hstu_fwd_kernel[grid_fill](
            q_f, k_f, v_f, u_f, mask_fill, o_f,
            q_f.stride(0), q_f.stride(1), q_f.stride(2), q_f.stride(3),
            k_f.stride(0), k_f.stride(1), k_f.stride(2), k_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),
            u_f.stride(0), u_f.stride(1), u_f.stride(2), u_f.stride(3),
            o_f.stride(0), o_f.stride(1), o_f.stride(2), o_f.stride(3),
            mask_fill.stride(0), mask_fill.stride(1), mask_fill.stride(2),
            B, H, N_q_fill, N_q_fill, float(cache_sz), HEAD_DIM=D, BLOCK_DMODEL=D
        )
        
        k_cache[:, :, :, :] = k_fill
        v_cache[:, :, :, :] = v_fill
        output[:, :cache_sz, :, :] = o_f.transpose(1, 2)
        
        # --- Decoding Phase (Sliding Window) ---
        L_total = q.shape[1]
        for offset in range(cache_sz, L_total, micro_bs):
            batch_end = min(offset + micro_bs, L_total)
            
            q_decode_orig = q[:, offset:batch_end, :, :]
            k_new = k[:, offset:batch_end, :, :]
            v_new = v[:, offset:batch_end, :, :]
            u_decode_orig = u[:, offset:batch_end, :, :]
            
            # Slide the cache window
            k_cache[:, -k_new.size(1):, :, :] = k_new
            v_cache[:, -v_new.size(1):, :, :] = v_new
            
            # The mask needs to be for the new q attending to the full cache
            mask_decode = torch.cat((mask[:, offset:batch_end, :user_length],mask[:,offset:batch_end,offset:batch_end]),dim=2)

            B_d, N_q_decode, H_d, D_d = q_decode_orig.shape
            _, N_kv_decode, _, _ = k_cache.shape

            q_d = q_decode_orig.transpose(1, 2).contiguous()
            k_d = k_cache.transpose(1, 2).contiguous()
            v_d = v_cache.transpose(1, 2).contiguous()
            u_d = u_decode_orig.transpose(1, 2).contiguous()
            o_d = torch.empty_like(q_d)

            grid_decode = lambda meta: (triton.cdiv(N_q_decode, meta['BLOCK_M']), B_d * H_d)

            _hstu_fwd_kernel[grid_decode](
                q_d, k_d, v_d, u_d, mask_decode, o_d,
                q_d.stride(0), q_d.stride(1), q_d.stride(2), q_d.stride(3),
                k_d.stride(0), k_d.stride(1), k_d.stride(2), k_d.stride(3),
                v_d.stride(0), v_d.stride(1), v_d.stride(2), v_d.stride(3),
                u_d.stride(0), u_d.stride(1), u_d.stride(2), u_d.stride(3),
                o_d.stride(0), o_d.stride(1), o_d.stride(2), o_d.stride(3),
                mask_decode.stride(0), mask_decode.stride(1), mask_decode.stride(2),
                B_d, H_d, N_q_decode, N_kv_decode, float(cache_sz), HEAD_DIM=D_d, BLOCK_DMODEL=D_d
            )
            output[:, offset:batch_end, :, :] = o_d.transpose(1, 2)
        
        return output

    # Allocate fixed-size KV cache for the sliding window
    cache_size = user_length + micorbatch_size
    k_cache = torch.zeros((B, cache_size, H, D), device='cuda', dtype=dtype)
    v_cache = torch.zeros((B, cache_size, H, D), device='cuda', dtype=dtype)

    print("\n--- Compiling Sliding Window KV Cache loop with torch.compile ---")
    _ = compiled_sliding_window_kv(q, k, v, u, mask, k_cache, v_cache, cache_size, micorbatch_size, config_kwargs)
    torch.cuda.synchronize()
    print("Compilation complete.")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(num_repeat):
        _  = compiled_sliding_window_kv(q, k, v, u, mask, k_cache, v_cache, cache_size, micorbatch_size, config_kwargs)
    torch.cuda.synchronize()
    end_time = time.time()

    time_with_kv = (end_time - start_time) / num_repeat
    peak_mem_with_kv = torch.cuda.max_memory_allocated() / 1e6 # in MB
    tflops_with_kv = flops_with_kv / time_with_kv / 1e12

    print("\n--- Analysis: With Sliding Window KV Cache (Compiled) ---")
    print(f"  Avg. Time:            {time_with_kv:.6f} s")
    print(f"  Peak Memory:          {peak_mem_with_kv:.2f} MB")
    print(f"  Theoretical FLOPs:    {flops_with_kv / 1e12:.4f} TFLOPs")
    print(f"  Achieved Performance: {tflops_with_kv:.4f} TFLOP/s")

    # --- Final Summary ---
    print("\n--- Summary of KV Cache Benefits ---")
    print(f"  Speedup:              {time_no_kv / time_with_kv:.2f}x")
    print(f"  Memory Savings:       {peak_mem_no_kv - peak_mem_with_kv:.2f} MB ({100 * (peak_mem_no_kv - peak_mem_with_kv) / peak_mem_no_kv:.2f}% reduction)")


if __name__ == "__main__":
    # To run this test, you need to have Triton installed (`pip install triton`)
    # and a compatible GPU with PyTorch 2.0+
    test_hstu_attention_long_target_triton(num_repeat=10)