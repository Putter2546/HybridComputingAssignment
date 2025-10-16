#!/usr/bin/env python3

import pyopencl as cl
import numpy as np
import time

# OpenCL Kernel Code สำหรับการคูณ Matrix
kernel_code = """
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

// Optimized version ใช้ Local Memory (Shared Memory)
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
        // Load tile into local memory
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
            sum += Asub[localRow * TILE_SIZE + k] * 
                   Bsub[k * TILE_SIZE + localCol];
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
    print("=" * 60)
    print("  PyOpenCL Matrix Multiplication")
    print("=" * 60)
    
    # ขนาด Matrix (N x N)
    N = 1024
    print(f"\nMatrix Size: {N} x {N}")
    
    # สร้าง OpenCL Context และ Queue
    print("\n--- Initializing OpenCL ---")
    
    # แสดง Platforms ที่มีทั้งหมด
    platforms = cl.get_platforms()
    print(f"\nAvailable Platforms: {len(platforms)}")
    for i, platform in enumerate(platforms):
        print(f"  [{i}] {platform.name} - {platform.vendor}")
    
    # เลือก Platform แรก (หรือระบุเองได้)
    platform = platforms[0]
    print(f"\nUsing Platform: {platform.name}")
    
    # แสดง Devices ทั้งหมด
    devices = platform.get_devices()
    print(f"\nAvailable Devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device.name} ({cl.device_type.to_string(device.type)})")
    
    # เลือก Device (GPU ถ้ามี, ไม่งั้นใช้ CPU)
    try:
        device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    except:
        device = platform.get_devices(device_type=cl.device_type.CPU)[0]
    
    print(f"\nUsing Device: {device.name}")
    
    # แสดงข้อมูล Device
    device_info = get_opencl_device_info(device)
    print("\nDevice Information:")
    print(f"  Type: {device_info['type']}")
    print(f"  Vendor: {device_info['vendor']}")
    print(f"  Compute Units: {device_info['max_compute_units']}")
    print(f"  Max Work Group Size: {device_info['max_work_group_size']}")
    print(f"  Clock Frequency: {device_info['max_clock_frequency']} MHz")
    print(f"  Global Memory: {device_info['global_mem_size']:.2f} GB")
    print(f"  Local Memory: {device_info['local_mem_size']:.2f} KB")
    
    # สร้าง Context และ Command Queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Compile OpenCL Program
    print("\n--- Compiling OpenCL Kernel ---")
    program = cl.Program(context, kernel_code).build()
    print("Kernel compiled successfully!")
    
    # สร้าง Matrix A และ B (random values)
    print(f"\n--- Preparing Data ---")
    np.random.seed(42)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    
    print(f"Matrix A: {A.shape}, dtype={A.dtype}")
    print(f"Matrix B: {B.shape}, dtype={B.dtype}")
    print(f"Memory per matrix: {A.nbytes / (1024**2):.2f} MB")
    
    # สร้าง OpenCL Buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
    
    # กำหนดขนาด Work-group
    local_size = (16, 16)  # 16x16 threads per work-group
    global_size = (
        ((N + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
    )
    
    print(f"\nWork-group configuration:")
    print(f"  Local size: {local_size}")
    print(f"  Global size: {global_size}")
    print(f"  Total work-items: {global_size[0] * global_size[1]}")
    
    # รัน Kernel พร้อมวัดเวลา
    print("\n--- Executing on GPU ---")
    
    # Basic version
    event = program.matrixMul(queue, global_size, local_size,
                             a_buf, b_buf, c_buf, np.int32(N))
    event.wait()
    
    # วัดเวลาจาก Event
    gpu_time = (event.profile.end - event.profile.start) * 1e-6  # แปลงเป็น ms
    
    # คัดลอกผลลัพธ์กลับมา
    cl.enqueue_copy(queue, C, c_buf).wait()
    
    print(f"GPU Execution Time: {gpu_time:.3f} ms")
    
    # คำนวณบน CPU เพื่อเปรียบเทียบ
    print("\n--- Executing on CPU ---")
    cpu_start = time.time()
    C_cpu = matrix_multiply_cpu(A, B)
    cpu_time = (time.time() - cpu_start) * 1000  # แปลงเป็น ms
    print(f"CPU Execution Time: {cpu_time:.3f} ms")
    
    # ตรวจสอบความถูกต้อง
    print("\n--- Verification ---")
    max_diff = np.max(np.abs(C - C_cpu))
    avg_diff = np.mean(np.abs(C - C_cpu))
    
    tolerance = 1e-3
    is_correct = max_diff < tolerance
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Average difference: {avg_diff:.6e}")
    print(f"Result: {'PASSED ✓' if is_correct else 'FAILED ✗'}")
    
    # แสดงผลลัพธ์
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"Matrix Size: {N} x {N}")
    print(f"GPU Time: {gpu_time:.3f} ms")
    print(f"CPU Time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    
    # คำนวณ GFLOPS
    operations = 2 * N * N * N  # 2N³ operations
    gflops = operations / (gpu_time * 1e6)
    print(f"GPU Performance: {gflops:.2f} GFLOPS")
    
    # คำนวณ Bandwidth
    bytes_transferred = 3 * N * N * 4  # 3 matrices * 4 bytes/float
    bandwidth = bytes_transferred / (gpu_time * 1e6)  # GB/s
    print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")
    
    print("=" * 60)
    
    # ตัวอย่างค่าใน Matrix
    print("\nSample values (top-left 3x3):")
    print("Matrix A:")
    print(A[:3, :3])
    print("\nMatrix B:")
    print(B[:3, :3])
    print("\nMatrix C (GPU):")
    print(C[:3, :3])
    print("\nMatrix C (CPU):")
    print(C_cpu[:3, :3])
    
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
        