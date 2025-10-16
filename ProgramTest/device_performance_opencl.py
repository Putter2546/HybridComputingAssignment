import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# ขนาดของข้อมูลที่จะใช้ทดสอบ
N = 5_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
results = {}

# สร้าง Kernel สำหรับบวกเวกเตอร์ (Vector Addition)
kernel_code = """
__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# ดึง Platform และ Device ที่มีในเครื่อง
for platform in cl.get_platforms():
    print(f"\nPlatform: {platform.name}")
    for device in platform.get_devices():
        print(f"  Testing device: {device.name}")
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        # เตรียม Buffer และ Compile Kernel
        program = cl.Program(context, kernel_code).build()
        mf = cl.mem_flags
        a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)

        # รันและจับเวลา
        start = time.time()
        program.vector_add(queue, (N,), None, a_buf, b_buf, c_buf)
        queue.finish()
        end = time.time()
        exec_time = end - start

        # เก็บผลลัพธ์
        results[device.name] = exec_time
        print(f"    Execution Time: {exec_time:.5f} seconds")

# ---------- แสดงกราฟเปรียบเทียบ ----------
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison of CPU/GPU Devices')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()
