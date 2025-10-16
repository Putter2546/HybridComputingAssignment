import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# ขนาดของข้อมูล
N = 10_000_000
a = np.random.rand(N).astype(np.float32)

# ---------- CPU (Sequential) ----------
start_cpu = time.time()
sum_cpu = np.sum(a)
end_cpu = time.time()
time_cpu = end_cpu - start_cpu

# ---------- GPU (Parallel with OpenCL) ----------
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Kernel สำหรับรวมค่าในแต่ละตำแหน่ง (แบบขนาน)
kernel_code = """
__kernel void reduce_sum(__global const float *a, __global float *partial_sums) {
    int gid = get_global_id(0);
    partial_sums[gid] = a[gid];
}
"""

program = cl.Program(context, kernel_code).build()

# สร้าง buffer
mf = cl.mem_flags
a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
partial_sums = np.empty_like(a)
partial_buf = cl.Buffer(context, mf.WRITE_ONLY, partial_sums.nbytes)

# รัน kernel
start_gpu = time.time()
program.reduce_sum(queue, (N,), None, a_buf, partial_buf)
cl.enqueue_copy(queue, partial_sums, partial_buf)
sum_gpu = np.sum(partial_sums)  # รวมผลลัพธ์จาก GPU
end_gpu = time.time()
time_gpu = end_gpu - start_gpu

# ---------- แสดงผล ----------
print(f"Sum (CPU): {sum_cpu:.4f}")
print(f"Sum (GPU): {sum_gpu:.4f}")
print(f"CPU Time: {time_cpu:.5f} s")
print(f"GPU Time: {time_gpu:.5f} s")

# ---------- แสดงกราฟเปรียบเทียบ ----------
methods = ['CPU (Sequential)', 'GPU (Parallel)']
times = [time_cpu, time_gpu]

plt.figure(figsize=(6,4))
plt.bar(methods, times, color=['gray', 'orange'])
plt.ylabel('Execution Time (seconds)')
plt.title('CPU vs GPU Parallel Processing')
plt.show()
