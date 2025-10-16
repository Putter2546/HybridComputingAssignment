#!/usr/bin/env python3
import os
os.environ["PYOPENCL_NO_CACHE"] = "1" 

import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------ เตรียมข้อมูล ------------------
N = 100_000_000
a = np.random.rand(N).astype(np.float32)

# ------------------ CPU (Sequential) ------------------
start_cpu = time.time()
sum_cpu = np.sum(a)
end_cpu = time.time()
time_cpu = end_cpu - start_cpu

# ------------------ GPU (Parallel Reduction) ------------------
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
mf = cl.mem_flags

# Kernel แบบ Parallel Reduction
kernel_code = """
__kernel void reduce_sum(__global const float *input, __global float *output, __local float *local_mem)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // โหลดข้อมูลจาก global memory → local memory
    local_mem[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction ภายใน work-group
    for (int stride = group_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            local_mem[lid] += local_mem[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // เขียนผลลัพธ์ของแต่ละ group กลับไปยัง global memory
    if (lid == 0)
        output[get_group_id(0)] = local_mem[0];
}
"""

program = cl.Program(context, kernel_code).build()

# ตั้งค่าขนาดการทำงาน
local_size = 256
num_groups = N // local_size
if N % local_size != 0:
    num_groups += 1

# เตรียม buffer
a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
partial_buf = cl.Buffer(context, mf.WRITE_ONLY, num_groups * np.float32().nbytes)

# เริ่มจับเวลา GPU
start_gpu = time.time()

# เรียกใช้ kernel ครั้งแรก
program.reduce_sum(
    queue,
    (num_groups * local_size,),
    (local_size,),
    a_buf,
    partial_buf,
    cl.LocalMemory(local_size * np.float32().nbytes)
)

# ดึงผลลัพธ์รอบแรก
partial_sums = np.empty(num_groups, dtype=np.float32)
cl.enqueue_copy(queue, partial_sums, partial_buf)

# รวมผลลัพธ์รอบสองบน CPU (เนื่องจากมีน้อยมาก)
sum_gpu = np.sum(partial_sums)
end_gpu = time.time()
time_gpu = end_gpu - start_gpu

# ------------------ แสดงผล ------------------
print(f"Sum (CPU): {sum_cpu:.4f}")
print(f"Sum (GPU): {sum_gpu:.4f}")
print(f"CPU Time: {time_cpu:.5f} s")
print(f"GPU Time: {time_gpu:.5f} s")

# ------------------ กราฟเปรียบเทียบ ------------------
methods = ['CPU (Sequential)', 'GPU (Parallel)']
times = [time_cpu, time_gpu]

plt.figure(figsize=(6,4))
plt.bar(methods, times, color=['gray', 'orange'])
plt.ylabel('Execution Time (seconds)')
plt.title('CPU vs GPU Parallel Reduction (PyOpenCL)')
plt.show()
