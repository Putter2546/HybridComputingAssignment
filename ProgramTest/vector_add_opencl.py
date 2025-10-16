import pyopencl as cl
import numpy as np
import time

# สร้างข้อมูลตัวอย่าง
N = 100000000
a = np.arange(N).astype(np.float32)
b = np.arange(N).astype(np.float32)
c = np.empty_like(a)

# เตรียม Platform และ Device
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# เขียน Kernel (โค้ดที่รันบน GPU)
kernel_code = """
__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# สร้างและ compile โปรแกรม
program = cl.Program(context, kernel_code).build()

# เตรียม memory buffer
mf = cl.mem_flags
a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(context, mf.WRITE_ONLY, c.nbytes)

# รัน kernel
program.vector_add(queue, a.shape, None, a_buf, b_buf, c_buf)
cl.enqueue_copy(queue, c, c_buf)

# วัดเวลาการรัน Kernel
start = time.time()
program.vector_add(queue, a.shape, None, a_buf, b_buf, c_buf)
queue.finish()
end = time.time()
gpu_time = end - start

print("Vector A:", a)
print("Vector B:", b)
print("Result (A + B):", c)

print("Execution Time:", end - start, "seconds")
print("Running on device:", device.name)
