#!/usr/bin/env python3

import pyopencl as cl
import numpy as np
import time

# OpenCL Kernel Code สำหรับการคูณ Matrix
kernel_code = """
// 1. Kernel พื้นฐาน (Naive)
__kernel void matrixMul(__global const float* A,
                       __global const float* B,
                       __global float* C,
                       const int N) {
    
    int row = get_global_id(1);  // Row index
    int col = get_global_id(0);  // Column index
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Kernel ที่ปรับปรุงแล้ว (Optimized) ใช้ Local Memory (Shared Memory)
__kernel void matrixMulOptimized(__global const float* A,
                                __global const float* B,
                                __global float* C,
                                const int N,
                                __local float* Asub,
                                __local float* Bsub) {
    
    const int TILE_SIZE = 16;
    int globalRow = get_global_id(1);
    int globalCol = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile into local memory (Shared Memory)
        int tiledRow = t * TILE_SIZE + localRow;
        int tiledCol = t * TILE_SIZE + localCol;
        
        Asub[localRow * TILE_SIZE + localCol] = 
            (globalRow < N && tiledCol < N) ? 
            A[globalRow * N + tiledCol] : 0.0f;
            
        Bsub[localRow * TILE_SIZE + localCol] = 
            (tiledRow < N && globalCol < N) ? 
            B[tiledRow * N + globalCol] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[localRow * TILE_SIZE + k] * Bsub[k * TILE_SIZE + localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (globalRow < N && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}
"""


def matrix_multiply_cpu(A, B):
    """Matrix multiplication บน CPU สำหรับเปรียบเทียบ"""
    return np.dot(A, B)


def get_opencl_device_info(device):
    """แสดงข้อมูล OpenCL Device"""
    info = {
        'name': device.name,
        'type': cl.device_type.to_string(device.type),
        'vendor': device.vendor,
        'version': device.version,
        'max_compute_units': device.max_compute_units,
        'max_work_group_size': device.max_work_group_size,
        'max_clock_frequency': device.max_clock_frequency,
        'global_mem_size': device.global_mem_size / (1024**3),  # GB
        'local_mem_size': device.local_mem_size / 1024,  # KB
    }
    return info


def main():
    print("=" * 70)
    print("       PyOpenCL Matrix Multiplication Benchmark")
    print("=" * 70)
    
    # ขนาด Matrix (N x N)
    N = 1024
    print(f"\nMatrix Size: {N} x {N}")
    
    # --- Initializing OpenCL ---
    
    platforms = cl.get_platforms()
    platform = platforms[0]
    
    # เลือก Device (GPU ถ้ามี, ไม่งั้นใช้ CPU)
    try:
        device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    except:
        device = platform.get_devices(device_type=cl.device_type.CPU)[0]
    
    print(f"\nUsing Device: {device.name} ({cl.device_type.to_string(device.type)})")
    
    # สร้าง Context และ Command Queue พร้อมเปิด Profiling
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Compile OpenCL Program
    program = cl.Program(context, kernel_code).build()
    
    # --- Preparing Data ---
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    
    # สร้าง OpenCL Buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
    
    # กำหนดขนาด Work-group (ต้องเป็นไปตาม TILE_SIZE=16)
    TILE_SIZE = 16
    local_size = (TILE_SIZE, TILE_SIZE)  # 16x16 threads per work-group
    global_size = (N, N)
    
    # Local memory size สำหรับ Asub และ Bsub (TILE_SIZE * TILE_SIZE * sizeof(float))
    local_mem_size = TILE_SIZE * TILE_SIZE * np.float32().nbytes 
    
    # --- 1. CPU Benchmark (NumPy) ---
    print("\n--- 1. CPU Execution (NumPy) ---")
    cpu_start = time.time()
    C_cpu = matrix_multiply_cpu(A, B)
    cpu_time = (time.time() - cpu_start) * 1000  # แปลงเป็น ms
    print(f"CPU Time: {cpu_time:.3f} ms")
    
    # --- 2. GPU Basic Kernel Benchmark ---
    print("\n--- 2. GPU Execution (Basic Kernel) ---")
    
    # รัน Kernel (Basic version)
    event_basic = program.matrixMul(queue, global_size, local_size,
                                  a_buf, b_buf, c_buf, np.int32(N))
    event_basic.wait()
    
    # วัดเวลา
    gpu_basic_time = (event_basic.profile.end - event_basic.profile.start) * 1e-6
    print(f"GPU Basic Time: {gpu_basic_time:.3f} ms")
    
    # --- 3. GPU Optimized Kernel Benchmark (Tiling/Local Memory) ---
    print("\n--- 3. GPU Execution (Optimized Kernel) ---")
    
    # รัน Kernel (Optimized version)
    event_opt = program.matrixMulOptimized(
        queue, 
        global_size, 
        local_size, 
        a_buf, 
        b_buf, 
        c_buf, 
        np.int32(N),
        cl.LocalMemory(local_mem_size), # Asub
        cl.LocalMemory(local_mem_size)  # Bsub
    )
    event_opt.wait()
    
    # วัดเวลา
    gpu_opt_time = (event_opt.profile.end - event_opt.profile.start) * 1e-6
    print(f"GPU Optimized Time: {gpu_opt_time:.3f} ms")
    
    # คัดลอกผลลัพธ์กลับมา (ใช้ผลลัพธ์จาก Optimized)
    cl.enqueue_copy(queue, C, c_buf).wait()

    # --- Verification (ใช้ผลลัพธ์จาก Optimized เทียบกับ CPU) ---
    print("\n--- Verification ---")
    max_diff = np.max(np.abs(C - C_cpu))
    tolerance = 1e-3
    is_correct = max_diff < tolerance
    print(f"Max difference: {max_diff:.6e}")
    print(f"Result: {'PASSED ✓' if is_correct else 'FAILED ✗'}")
    
    # --- Results Summary ---
    print("\n" + "=" * 70)
    print("           FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Matrix Size: {N} x {N}")
    print("-" * 35)
    print(f"1. CPU Time (NumPy): {cpu_time:.3f} ms")
    print(f"2. GPU Basic Time:   {gpu_basic_time:.3f} ms")
    print(f"3. GPU Optimized Time: {gpu_opt_time:.3f} ms")
    print("-" * 35)
    print(f"Speedup (Optimized vs CPU): {cpu_time / gpu_opt_time:.2f}x")
    print(f"Optimization Gain (3 vs 2): {gpu_basic_time / gpu_opt_time:.2f}x")
    print("=" * 70)
    
    # Cleanup
    a_buf.release()
    b_buf.release()
    c_buf.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. ตรวจสอบว่าติดตั้ง PyOpenCL แล้ว: pip install pyopencl")
        print("2. ตรวจสอบว่ามี OpenCL runtime: clinfo")
        print("3. ตรวจสอบ GPU driver รุ่นล่าสุด")
        import traceback
        traceback.print_exc()
