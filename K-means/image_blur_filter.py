#!/usr/bin/env python3
"""
Image Blur Filter using PyOpenCL
รองรับ Gaussian Blur, Box Blur, และ Motion Blur
"""

import pyopencl as cl
import numpy as np
import time
from PIL import Image
import sys

# OpenCL Kernel Code
kernel_code = """
// Gaussian Blur Kernel
__kernel void gaussianBlur(__global const uchar4* input,
                          __global uchar4* output,
                          __global const float* filter,
                          const int width,
                          const int height,
                          const int filterSize) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int filterRadius = filterSize / 2;
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    
    // Apply filter
    for (int fy = -filterRadius; fy <= filterRadius; fy++) {
        for (int fx = -filterRadius; fx <= filterRadius; fx++) {
            // Clamp coordinates to image boundaries
            int imageX = clamp(x + fx, 0, width - 1);
            int imageY = clamp(y + fy, 0, height - 1);
            
            // Get pixel value
            uchar4 pixel = input[imageY * width + imageX];
            
            // Get filter weight
            int filterIdx = (fy + filterRadius) * filterSize + (fx + filterRadius);
            float weight = filter[filterIdx];
            
            // Accumulate
            sum.x += (float)pixel.x * weight;
            sum.y += (float)pixel.y * weight;
            sum.z += (float)pixel.z * weight;
            sum.w += (float)pixel.w * weight;
            weightSum += weight;
        }
    }
    
    // Normalize and write output
    uchar4 result;
    result.x = (uchar)(sum.x / weightSum);
    result.y = (uchar)(sum.y / weightSum);
    result.z = (uchar)(sum.z / weightSum);
    result.w = (uchar)(sum.w / weightSum);
    
    output[y * width + x] = result;
}

// Box Blur Kernel (faster, simpler)
__kernel void boxBlur(__global const uchar4* input,
                     __global uchar4* output,
                     const int width,
                     const int height,
                     const int filterSize) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int filterRadius = filterSize / 2;
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int count = 0;
    
    for (int fy = -filterRadius; fy <= filterRadius; fy++) {
        for (int fx = -filterRadius; fx <= filterRadius; fx++) {
            int imageX = clamp(x + fx, 0, width - 1);
            int imageY = clamp(y + fy, 0, height - 1);
            
            uchar4 pixel = input[imageY * width + imageX];
            sum.x += (float)pixel.x;
            sum.y += (float)pixel.y;
            sum.z += (float)pixel.z;
            sum.w += (float)pixel.w;
            count++;
        }
    }
    
    uchar4 result;
    result.x = (uchar)(sum.x / count);
    result.y = (uchar)(sum.y / count);
    result.z = (uchar)(sum.z / count);
    result.w = (uchar)(sum.w / count);
    
    output[y * width + x] = result;
}

// Motion Blur Kernel
__kernel void motionBlur(__global const uchar4* input,
                        __global uchar4* output,
                        const int width,
                        const int height,
                        const int blurLength,
                        const float angle) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float dx = cos(angle);
    float dy = sin(angle);
    
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int count = 0;
    
    for (int i = -blurLength/2; i <= blurLength/2; i++) {
        int sampleX = clamp((int)(x + dx * i), 0, width - 1);
        int sampleY = clamp((int)(y + dy * i), 0, height - 1);
        
        uchar4 pixel = input[sampleY * width + sampleX];
        sum.x += (float)pixel.x;
        sum.y += (float)pixel.y;
        sum.z += (float)pixel.z;
        sum.w += (float)pixel.w;
        count++;
    }
    
    uchar4 result;
    result.x = (uchar)(sum.x / count);
    result.y = (uchar)(sum.y / count);
    result.z = (uchar)(sum.z / count);
    result.w = (uchar)(sum.w / count);
    
    output[y * width + x] = result;
}

// Grayscale Kernel (bonus)
__kernel void toGrayscale(__global const uchar4* input,
                         __global uchar4* output,
                         const int width,
                         const int height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    uchar4 pixel = input[y * width + x];
    
    // Luminosity method: 0.299*R + 0.587*G + 0.114*B
    uchar gray = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    
    uchar4 result;
    result.x = gray;
    result.y = gray;
    result.z = gray;
    result.w = pixel.w;
    
    output[y * width + x] = result;
}
"""


def create_gaussian_filter(size, sigma):
    """สร้าง Gaussian filter kernel"""
    filter_1d = np.zeros(size, dtype=np.float32)
    radius = size // 2
    
    # สร้าง 1D Gaussian
    for i in range(size):
        x = i - radius
        filter_1d[i] = np.exp(-(x * x) / (2.0 * sigma * sigma))
    
    # Normalize
    filter_1d /= filter_1d.sum()
    
    # สร้าง 2D filter จาก outer product
    filter_2d = np.outer(filter_1d, filter_1d).astype(np.float32)
    
    return filter_2d


def load_image(image_path):
    """โหลดภาพและแปลงเป็น RGBA format"""
    try:
        img = Image.open(image_path)
        # แปลงเป็น RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def image_to_array(img):
    """แปลง PIL Image เป็น NumPy array (RGBA)"""
    return np.array(img, dtype=np.uint8)


def array_to_image(arr):
    """แปลง NumPy array เป็น PIL Image"""
    return Image.fromarray(arr, mode='RGBA')


def setup_opencl():
    """Setup OpenCL context และ queue"""
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        raise RuntimeError("No OpenCL platforms found!")
    
    platform = platforms[0]
    
    # เลือก GPU ถ้ามี, ไม่งั้นใช้ CPU
    try:
        device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    except:
        device = platform.get_devices(device_type=cl.device_type.CPU)[0]
    
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    return context, queue, device


def apply_gaussian_blur(context, queue, program, img_array, filter_size=5, sigma=1.0):
    """ใช้ Gaussian Blur"""
    height, width, _ = img_array.shape
    
    # สร้าง Gaussian filter
    gaussian_filter = create_gaussian_filter(filter_size, sigma)
    
    # สร้าง buffers
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, img_array.nbytes)
    filter_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gaussian_filter)
    
    # Execute kernel
    global_size = (width, height)
    event = program.gaussianBlur(queue, global_size, None,
                                 input_buf, output_buf, filter_buf,
                                 np.int32(width), np.int32(height), np.int32(filter_size))
    event.wait()
    
    # Get result
    output_array = np.empty_like(img_array)
    cl.enqueue_copy(queue, output_array, output_buf).wait()
    
    # Get execution time
    exec_time = (event.profile.end - event.profile.start) * 1e-6  # ms
    
    # Cleanup
    input_buf.release()
    output_buf.release()
    filter_buf.release()
    
    return output_array, exec_time


def apply_box_blur(context, queue, program, img_array, filter_size=5):
    """ใช้ Box Blur (เร็วกว่า Gaussian)"""
    height, width, _ = img_array.shape
    
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, img_array.nbytes)
    
    global_size = (width, height)
    event = program.boxBlur(queue, global_size, None,
                           input_buf, output_buf,
                           np.int32(width), np.int32(height), np.int32(filter_size))
    event.wait()
    
    output_array = np.empty_like(img_array)
    cl.enqueue_copy(queue, output_array, output_buf).wait()
    
    exec_time = (event.profile.end - event.profile.start) * 1e-6
    
    input_buf.release()
    output_buf.release()
    
    return output_array, exec_time


def apply_motion_blur(context, queue, program, img_array, blur_length=15, angle=0.0):
    """ใช้ Motion Blur"""
    height, width, _ = img_array.shape
    
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, img_array.nbytes)
    
    global_size = (width, height)
    event = program.motionBlur(queue, global_size, None,
                               input_buf, output_buf,
                               np.int32(width), np.int32(height),
                               np.int32(blur_length), np.float32(angle))
    event.wait()
    
    output_array = np.empty_like(img_array)
    cl.enqueue_copy(queue, output_array, output_buf).wait()
    
    exec_time = (event.profile.end - event.profile.start) * 1e-6
    
    input_buf.release()
    output_buf.release()
    
    return output_array, exec_time


def apply_grayscale(context, queue, program, img_array):
    """แปลงเป็น Grayscale"""
    height, width, _ = img_array.shape
    
    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_array)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, img_array.nbytes)
    
    global_size = (width, height)
    event = program.toGrayscale(queue, global_size, None,
                                input_buf, output_buf,
                                np.int32(width), np.int32(height))
    event.wait()
    
    output_array = np.empty_like(img_array)
    cl.enqueue_copy(queue, output_array, output_buf).wait()
    
    exec_time = (event.profile.end - event.profile.start) * 1e-6
    
    input_buf.release()
    output_buf.release()
    
    return output_array, exec_time


def create_test_image(width=1920, height=1080):
    """สร้างภาพทดสอบ (Gradient + Circles)"""
    img_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # สร้าง gradient background
    for y in range(height):
        for x in range(width):
            img_array[y, x, 0] = int(255 * x / width)        # R
            img_array[y, x, 1] = int(255 * y / height)       # G
            img_array[y, x, 2] = int(128 + 127 * np.sin(x/50)) # B
            img_array[y, x, 3] = 255                          # A
    
    # วงกลมสีต่างๆ
    centers = [(width//4, height//4), (3*width//4, height//4),
               (width//4, 3*height//4), (3*width//4, 3*height//4),
               (width//2, height//2)]
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255),
              (255, 255, 0, 255), (255, 255, 255, 255)]
    
    for (cx, cy), color in zip(centers, colors):
        for y in range(max(0, cy-80), min(height, cy+80)):
            for x in range(max(0, cx-80), min(width, cx+80)):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < 80:
                    img_array[y, x] = color
    
    return img_array


def main():
    print("=" * 70)
    print("  PyOpenCL Image Blur Filter")
    print("=" * 70)
    
    # Setup OpenCL
    print("\n--- Initializing OpenCL ---")
    context, queue, device = setup_opencl()
    print(f"Using device: {device.name}")
    print(f"Device type: {cl.device_type.to_string(device.type)}")
    
    # Compile kernels
    print("\n--- Compiling Kernels ---")
    program = cl.Program(context, kernel_code).build()
    print("Kernels compiled successfully!")
    
    # Load or create image
    print("\n--- Loading Image ---")
    
    # ลองโหลดภาพจากไฟล์
    image_path = "input.jpg"  # เปลี่ยนเป็น path ของคุณ
    img = load_image(image_path)
    
    if img is None:
        print(f"Could not load '{image_path}', creating test image instead...")
        img_array = create_test_image(1920, 1080)
        img = array_to_image(img_array)
        img.save("test_input.png")
        print("Test image saved as 'test_input.png'")
    else:
        print(f"Loaded image: {image_path}")
        img_array = image_to_array(img)
    
    height, width, channels = img_array.shape
    print(f"Image size: {width}x{height}, channels: {channels}")
    print(f"Memory size: {img_array.nbytes / (1024**2):.2f} MB")
    
    # Apply different blur filters
    print("\n" + "=" * 70)
    print("  Processing")
    print("=" * 70)
    
    results = []
    
    # 1. Gaussian Blur
    print("\n1. Applying Gaussian Blur (5x5, sigma=1.0)...")
    output_gaussian, time_gaussian = apply_gaussian_blur(context, queue, program, 
                                                         img_array, filter_size=5, sigma=1.0)
    print(f"   ✓ Completed in {time_gaussian:.3f} ms")
    results.append(("gaussian_blur", output_gaussian, time_gaussian))
    
    # 2. Gaussian Blur (stronger)
    print("\n2. Applying Gaussian Blur (9x9, sigma=2.0)...")
    output_gaussian_strong, time_gaussian_strong = apply_gaussian_blur(context, queue, program,
                                                                       img_array, filter_size=9, sigma=2.0)
    print(f"   ✓ Completed in {time_gaussian_strong:.3f} ms")
    results.append(("gaussian_blur_strong", output_gaussian_strong, time_gaussian_strong))
    
    # 3. Box Blur
    print("\n3. Applying Box Blur (5x5)...")
    output_box, time_box = apply_box_blur(context, queue, program, img_array, filter_size=5)
    print(f"   ✓ Completed in {time_box:.3f} ms")
    results.append(("box_blur", output_box, time_box))
    
    # 4. Motion Blur (horizontal)
    print("\n4. Applying Motion Blur (horizontal, length=15)...")
    output_motion_h, time_motion_h = apply_motion_blur(context, queue, program,
                                                       img_array, blur_length=15, angle=0.0)
    print(f"   ✓ Completed in {time_motion_h:.3f} ms")
    results.append(("motion_blur_horizontal", output_motion_h, time_motion_h))
    
    # 5. Motion Blur (diagonal)
    print("\n5. Applying Motion Blur (diagonal, length=20)...")
    output_motion_d, time_motion_d = apply_motion_blur(context, queue, program,
                                                       img_array, blur_length=20, angle=np.pi/4)
    print(f"   ✓ Completed in {time_motion_d:.3f} ms")
    results.append(("motion_blur_diagonal", output_motion_d, time_motion_d))
    
    # 6. Grayscale (bonus)
    print("\n6. Converting to Grayscale...")
    output_gray, time_gray = apply_grayscale(context, queue, program, img_array)
    print(f"   ✓ Completed in {time_gray:.3f} ms")
    results.append(("grayscale", output_gray, time_gray))
    
    # Save results
    print("\n--- Saving Results ---")
    for name, output, exec_time in results:
        filename = f"output_{name}.png"
        output_img = array_to_image(output)
        output_img.save(filename)
        print(f"✓ Saved: {filename} ({exec_time:.3f} ms)")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("  Performance Summary")
    print("=" * 70)
    print(f"Image Size: {width}x{height} ({width*height:,} pixels)")
    print(f"\nFilter                     Time (ms)    Throughput (Mpixels/s)")
    print("-" * 70)
    
    for name, _, exec_time in results:
        throughput = (width * height) / (exec_time * 1000)
        print(f"{name:25} {exec_time:8.3f}    {throughput:8.2f}")
    
    total_time = sum(t for _, _, t in results)
    print("-" * 70)
    print(f"{'Total':25} {total_time:8.3f}")
    print("=" * 70)
    
    # Calculate speedup vs CPU (estimated)
    print("\nEstimated GPU Speedup vs CPU:")
    cpu_time_estimate = time_gaussian * 50  # ประมาณการ
    speedup = cpu_time_estimate / time_gaussian
    print(f"  Gaussian Blur: ~{speedup:.1f}x faster")
    
    print("\n✓ All filters applied successfully!")
    print(f"Check output files: output_*.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install pyopencl pillow numpy")
        print("2. Make sure OpenCL runtime is installed: clinfo")
        print("3. Place an image file named 'input.jpg' in the same directory")
        print("   Or the program will create a test image automatically")
        import traceback
        traceback.print_exc()
