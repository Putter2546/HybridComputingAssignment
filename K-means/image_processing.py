import pyopencl as cl
import numpy as np
from PIL import Image

# Load image
img = Image.open('input.jpg').convert('L')
img_array = np.array(img, dtype=np.float32)

# Gaussian Blur Kernel
kernel_code = """
__kernel void gaussianBlur(__global const float* input,
                           __global float* output,
                           const int width,
                           const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float kernel[3][3] = {
        {1.0/16, 2.0/16, 1.0/16},
        {2.0/16, 4.0/16, 2.0/16},
        {1.0/16, 2.0/16, 1.0/16}
    };
    
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(x + dx, 0, width - 1);
            int ny = clamp(y + dy, 0, height - 1);
            sum += input[ny * width + nx] * kernel[dy + 1][dx + 1];
        }
    }
    
    output[y * width + x] = sum;
}
"""

# Setup OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Buffers
mf = cl.mem_flags
input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                     hostbuf=img_array)
output_buf = cl.Buffer(context, mf.WRITE_ONLY, img_array.nbytes)

# Execute
program = cl.Program(context, kernel_code).build()
height, width = img_array.shape
program.gaussianBlur(queue, (width, height), None,
                    input_buf, output_buf,
                    np.int32(width), np.int32(height))

# Get result
output_array = np.empty_like(img_array)
cl.enqueue_copy(queue, output_array, output_buf).wait()

# Save
output_img = Image.fromarray(output_array.astype(np.uint8))
output_img.save('output.jpg')
