import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_matrix_sizes():
    """Benchmark different matrix sizes"""
    
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    kernel_code = """
    __kernel void matrixMul(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const int N) {
        int row = get_global_id(1);
        int col = get_global_id(0);
        
        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    
    program = cl.Program(context, kernel_code).build()
    
    sizes = [128, 256, 512, 1024, 2048]
    gpu_times = []
    cpu_times = []
    speedups = []
    
    for N in sizes:
        print(f"\nTesting matrix size: {N}x{N}")
        
        # Prepare data
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)
        
        # GPU
        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)
        
        local_size = (16, 16)
        global_size = (
            ((N + local_size[0] - 1) // local_size[0]) * local_size[0],
            ((N + local_size[1] - 1) // local_size[1]) * local_size[1]
        )
        
        # Warm-up
        program.matrixMul(queue, global_size, local_size,
                         a_buf, b_buf, c_buf, np.int32(N))
        queue.finish()
        
        # Measure GPU time
        gpu_start = time.time()
        for _ in range(5):
            program.matrixMul(queue, global_size, local_size,
                             a_buf, b_buf, c_buf, np.int32(N))
        queue.finish()
        gpu_time = (time.time() - gpu_start) / 5 * 1000
        
        # Measure CPU time
        cpu_start = time.time()
        C_cpu = np.dot(A, B)
        cpu_time = (time.time() - cpu_start) * 1000
        
        speedup = cpu_time / gpu_time
        
        gpu_times.append(gpu_time)
        cpu_times.append(cpu_time)
        speedups.append(speedup)
        
        print(f"  GPU: {gpu_time:.2f} ms")
        print(f"  CPU: {cpu_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Cleanup
        a_buf.release()
        b_buf.release()
        c_buf.release()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.plot(sizes, gpu_times, 'b-o', label='GPU', linewidth=2)
    ax1.plot(sizes, cpu_times, 'r-s', label='CPU', linewidth=2)
    ax1.set_xlabel('Matrix Size (N x N)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Speedup
    ax2.plot(sizes, speedups, 'g-^', linewidth=2, markersize=8)
    ax2.set_xlabel('Matrix Size (N x N)')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('GPU Speedup vs CPU')
    ax2.grid(True)
    ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    print("\nBenchmark plot saved as 'benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    benchmark_matrix_sizes()
    