#!/usr/bin/env python3
"""
K-Means Clustering using PyOpenCL
รองรับ Image Segmentation และ Data Clustering
"""

import pyopencl as cl
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

# OpenCL Kernel Code
kernel_code = """
// Assignment Step: กำหนด cluster ให้แต่ละจุดข้อมูล
__kernel void assignClusters(__global const float* data,
                            __global const float* centroids,
                            __global int* assignments,
                            const int numPoints,
                            const int numDimensions,
                            const int k) {
    
    int idx = get_global_id(0);
    
    if (idx >= numPoints) return;
    
    float minDistance = INFINITY;
    int bestCluster = 0;
    
    // หาระยะทางไปยังแต่ละ centroid
    for (int c = 0; c < k; c++) {
        float distance = 0.0f;
        
        // คำนวณ Euclidean distance
        for (int d = 0; d < numDimensions; d++) {
            float diff = data[idx * numDimensions + d] - 
                        centroids[c * numDimensions + d];
            distance += diff * diff;
        }
        
        // เก็บ cluster ที่ใกล้ที่สุด
        if (distance < minDistance) {
            minDistance = distance;
            bestCluster = c;
        }
    }
    
    assignments[idx] = bestCluster;
}

// Update Step: คำนวณ centroid ใหม่
__kernel void updateCentroids(__global const float* data,
                             __global const int* assignments,
                             __global float* newCentroids,
                             __global int* counts,
                             const int numPoints,
                             const int numDimensions,
                             const int k) {
    
    int c = get_global_id(0);  // cluster index
    int d = get_global_id(1);  // dimension index
    
    if (c >= k || d >= numDimensions) return;
    
    float sum = 0.0f;
    int count = 0;
    
    // รวมค่าของทุกจุดใน cluster นี้
    for (int i = 0; i < numPoints; i++) {
        if (assignments[i] == c) {
            sum += data[i * numDimensions + d];
            count++;
        }
    }
    
    // คำนวณค่าเฉลี่ย
    if (count > 0) {
        newCentroids[c * numDimensions + d] = sum / count;
        if (d == 0) {
            counts[c] = count;
        }
    } else {
        // ถ้าไม่มีจุดใน cluster นี้ ให้ใช้ค่าเดิม
        newCentroids[c * numDimensions + d] = 0.0f;
    }
}

// Image Segmentation: กำหนดสีให้แต่ละ pixel ตาม cluster
__kernel void colorByCluster(__global const int* assignments,
                            __global const float* centroids,
                            __global uchar4* output,
                            const int width,
                            const int height,
                            const int k) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int cluster = assignments[idx];
    
    // ใช้สีจาก centroid
    uchar4 color;
    color.x = (uchar)centroids[cluster * 3 + 0];
    color.y = (uchar)centroids[cluster * 3 + 1];
    color.z = (uchar)centroids[cluster * 3 + 2];
    color.w = 255;
    
    output[idx] = color;
}

// Compute Within-Cluster Sum of Squares (WCSS)
__kernel void computeWCSS(__global const float* data,
                         __global const float* centroids,
                         __global const int* assignments,
                         __global float* wcss,
                         const int numPoints,
                         const int numDimensions,
                         const int k) {
    
    int idx = get_global_id(0);
    
    if (idx >= numPoints) return;
    
    int cluster = assignments[idx];
    float distance = 0.0f;
    
    for (int d = 0; d < numDimensions; d++) {
        float diff = data[idx * numDimensions + d] - 
                    centroids[cluster * numDimensions + d];
        distance += diff * diff;
    }
    
    // Atomic add (สำหรับ GPU ที่รองรับ)
    // wcss[cluster] += distance;
    
    // Alternative: เก็บค่าแยกแล้วรวมทีหลัง
    wcss[idx] = distance;
}
"""


def setup_opencl():
    """Setup OpenCL context และ queue"""
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        raise RuntimeError("No OpenCL platforms found!")
    
    platform = platforms[0]
    
    try:
        device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    except:
        device = platform.get_devices(device_type=cl.device_type.CPU)[0]
    
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    return context, queue, device


def initialize_centroids(data, k, method='random'):
    """เริ่มต้น centroids"""
    n_samples, n_features = data.shape
    
    if method == 'random':
        # สุ่มจุดข้อมูล k จุดมาเป็น centroid
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices].copy()
    
    elif method == 'kmeans++':
        # K-Means++ initialization (ดีกว่า random)
        centroids = np.zeros((k, n_features), dtype=np.float32)
        
        # เลือก centroid แรกแบบสุ่ม
        centroids[0] = data[np.random.randint(n_samples)]
        
        for i in range(1, k):
            # คำนวณระยะทางของแต่ละจุดไปยัง centroid ที่ใกล้ที่สุด
            distances = np.min([
                np.sum((data - c)**2, axis=1) 
                for c in centroids[:i]
            ], axis=0)
            
            # เลือกจุดถัดไปโดยมีความน่าจะเป็นตามระยะทาง
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = data[j]
                    break
    
    return centroids.astype(np.float32)


def kmeans_opencl(data, k, max_iterations=100, tolerance=1e-4, init_method='kmeans++'):
    """
    K-Means Clustering ด้วย OpenCL
    
    Parameters:
    - data: numpy array (n_samples, n_features)
    - k: จำนวน clusters
    - max_iterations: จำนวนรอบสูงสุด
    - tolerance: ค่า threshold สำหรับการหยุด
    - init_method: 'random' หรือ 'kmeans++'
    
    Returns:
    - centroids: ตำแหน่งของ centroids
    - assignments: cluster ของแต่ละจุด
    - history: ประวัติการทำงาน
    """
    
    print(f"Starting K-Means with k={k}, init={init_method}")
    
    # Setup OpenCL
    context, queue, device = setup_opencl()
    print(f"Using device: {device.name}")
    
    # Compile program
    program = cl.Program(context, kernel_code).build()
    
    n_samples, n_features = data.shape
    print(f"Data: {n_samples} samples, {n_features} features")
    
    # Initialize centroids
    centroids = initialize_centroids(data, k, method=init_method)
    assignments = np.zeros(n_samples, dtype=np.int32)
    new_centroids = np.zeros((k, n_features), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int32)
    
    # Create OpenCL buffers
    mf = cl.mem_flags
    data_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    centroids_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=centroids)
    assignments_buf = cl.Buffer(context, mf.READ_WRITE, assignments.nbytes)
    new_centroids_buf = cl.Buffer(context, mf.READ_WRITE, new_centroids.nbytes)
    counts_buf = cl.Buffer(context, mf.READ_WRITE, counts.nbytes)
    
    history = {
        'iterations': [],
        'wcss': [],
        'times': [],
        'centroid_shifts': []
    }
    
    print("\n" + "="*60)
    print("Iteration | WCSS      | Shift     | Time (ms)")
    print("-"*60)
    
    total_time = 0
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Assignment Step
        global_size_assign = (n_samples,)
        event1 = program.assignClusters(
            queue, global_size_assign, None,
            data_buf, centroids_buf, assignments_buf,
            np.int32(n_samples), np.int32(n_features), np.int32(k)
        )
        event1.wait()
        
        # Read assignments
        cl.enqueue_copy(queue, assignments, assignments_buf).wait()
        
        # Update Step
        global_size_update = (k, n_features)
        event2 = program.updateCentroids(
            queue, global_size_update, None,
            data_buf, assignments_buf, new_centroids_buf, counts_buf,
            np.int32(n_samples), np.int32(n_features), np.int32(k)
        )
        event2.wait()
        
        # Read new centroids
        cl.enqueue_copy(queue, new_centroids, new_centroids_buf).wait()
        cl.enqueue_copy(queue, counts, counts_buf).wait()
        
        # คำนวณการเปลี่ยนแปลงของ centroids
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        
        # Update centroids
        centroids = new_centroids.copy()
        cl.enqueue_copy(queue, centroids_buf, centroids, is_blocking=True)
        
        # คำนวณ WCSS (Within-Cluster Sum of Squares)
        wcss = 0.0
        for i in range(k):
            cluster_points = data[assignments == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i])**2)
        
        iter_time = (time.time() - iter_start) * 1000
        total_time += iter_time
        
        # บันทึกประวัติ
        history['iterations'].append(iteration)
        history['wcss'].append(wcss)
        history['times'].append(iter_time)
        history['centroid_shifts'].append(centroid_shift)
        
        print(f"{iteration:9d} | {wcss:9.2f} | {centroid_shift:9.6f} | {iter_time:8.3f}")
        
        # Check convergence
        if centroid_shift < tolerance:
            print(f"\nConverged at iteration {iteration}")
            break
    
    print("="*60)
    print(f"Total time: {total_time:.2f} ms")
    print(f"Average time per iteration: {total_time/(iteration+1):.2f} ms")
    
    # Cleanup
    data_buf.release()
    centroids_buf.release()
    assignments_buf.release()
    new_centroids_buf.release()
    counts_buf.release()
    
    return centroids, assignments, history


def kmeans_image_segmentation(image_path, k=5, init_method='kmeans++'):
    """ใช้ K-Means สำหรับ Image Segmentation"""
    
    print("\n" + "="*70)
    print("  K-Means Image Segmentation")
    print("="*70)
    
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img, dtype=np.uint8)
    height, width, channels = img_array.shape
    
    print(f"Image: {width}x{height} ({width*height:,} pixels)")
    
    # แปลงเป็น data สำหรับ clustering (flatten)
    data = img_array.reshape(-1, 3).astype(np.float32)
    
    # Run K-Means
    centroids, assignments, history = kmeans_opencl(
        data, k, max_iterations=50, init_method=init_method
    )
    
    # สร้างภาพที่ segment แล้ว
    segmented = centroids[assignments].reshape(height, width, 3).astype(np.uint8)
    
    # Setup OpenCL สำหรับ visualization
    context, queue, device = setup_opencl()
    program = cl.Program(context, kernel_code).build()
    
    # Create output with distinct colors
    output_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    mf = cl.mem_flags
    assignments_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                               hostbuf=assignments.astype(np.int32))
    centroids_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                             hostbuf=centroids.astype(np.float32))
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, output_array.nbytes)
    
    program.colorByCluster(queue, (width, height), None,
                          assignments_buf, centroids_buf, output_buf,
                          np.int32(width), np.int32(height), np.int32(k))
    
    cl.enqueue_copy(queue, output_array, output_buf).wait()
    
    # Cleanup
    assignments_buf.release()
    centroids_buf.release()
    output_buf.release()
    
    return img_array, segmented, output_array, centroids, assignments, history


def visualize_results(original, segmented, colored, centroids, history, k):
    """แสดงผลลัพธ์"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Segmented image (using centroid colors)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(segmented)
    ax2.set_title(f'Segmented (k={k})', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Colored by cluster
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(colored[:,:,:3])
    ax3.set_title('Cluster Visualization', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # WCSS over iterations
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(history['iterations'], history['wcss'], 'b-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('WCSS')
    ax4.set_title('Within-Cluster Sum of Squares')
    ax4.grid(True, alpha=0.3)
    
    # Centroid shift over iterations
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(history['iterations'], history['centroid_shifts'], 'r-', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Centroid Shift')
    ax5.set_title('Convergence')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Color palette (centroids)
    ax6 = plt.subplot(2, 3, 6)
    colors = centroids / 255.0
    for i in range(k):
        ax6.add_patch(plt.Rectangle((i, 0), 1, 1, 
                                    facecolor=colors[i], 
                                    edgecolor='black', linewidth=2))
    ax6.set_xlim(0, k)
    ax6.set_ylim(0, 1)
    ax6.set_aspect('equal')
    ax6.set_title(f'Color Palette ({k} clusters)')
    ax6.set_xticks(range(k))
    ax6.set_xticklabels([f'C{i}' for i in range(k)])
    ax6.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'kmeans_result_k{k}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: kmeans_result_k{k}.png")
    plt.show()


def elbow_method(data, max_k=10):
    """Elbow Method เพื่อหาค่า k ที่เหมาะสม"""
    
    print("\n" + "="*60)
    print("  Elbow Method - Finding Optimal k")
    print("="*60)
    
    wcss_list = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        print(f"\nTesting k={k}...")
        _, _, history = kmeans_opencl(data, k, max_iterations=30, init_method='kmeans++')
        final_wcss = history['wcss'][-1]
        wcss_list.append(final_wcss)
        print(f"k={k}: WCSS = {final_wcss:.2f}")
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), wcss_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares', fontsize=12)
    plt.title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(k_range))
    
    plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
    print("\n✓ Elbow curve saved: elbow_curve.png")
    plt.show()
    
    return wcss_list


def create_test_data_2d(n_samples=1000, n_clusters=3):
    """สร้างข้อมูลทดสอบ 2D"""
    np.random.seed(42)
    
    data = []
    labels = []
    
    # สร้าง Gaussian clusters
    cluster_centers = np.random.rand(n_clusters, 2) * 100
    
    for i in range(n_clusters):
        cluster_data = np.random.randn(n_samples // n_clusters, 2) * 10 + cluster_centers[i]
        data.append(cluster_data)
        labels.extend([i] * (n_samples // n_clusters))
    
    data = np.vstack(data).astype(np.float32)
    labels = np.array(labels)
    
    return data, labels, cluster_centers


def demo_2d_clustering():
    """Demo K-Means บนข้อมูล 2D"""
    
    print("\n" + "="*70)
    print("  K-Means 2D Clustering Demo")
    print("="*70)
    
    # สร้างข้อมูลทดสอบ
    k_true = 4
    data, true_labels, true_centers = create_test_data_2d(n_samples=2000, n_clusters=k_true)
    
    # Run K-Means
    centroids, assignments, history = kmeans_opencl(
        data, k=k_true, max_iterations=50, init_method='kmeans++'
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original data with true labels
    axes[0].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', 
                   alpha=0.6, s=20)
    axes[0].scatter(true_centers[:, 0], true_centers[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='True Centers')
    axes[0].set_title('True Clusters', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # K-Means result
    axes[1].scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis',
                   alpha=0.6, s=20)
    axes[1].scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='K-Means Centroids')
    axes[1].set_title(f'K-Means Clustering (k={k_true})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Convergence
    axes[2].plot(history['iterations'], history['wcss'], 'b-o', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('WCSS')
    axes[2].set_title('Convergence', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_2d_demo.png', dpi=150)
    print("\n✓ 2D clustering visualization saved: kmeans_2d_demo.png")
    plt.show()


def main():
    """Main function"""
    
    print("=" * 70)
    print("  PyOpenCL K-Means Clustering")
    print("="* 70)
    
    # เลือกโหมด
    print("\nSelect mode:")
    print("1. Image Segmentation")
    print("2. 2D Data Clustering Demo")
    print("3. Elbow Method (find optimal k)")
    
    choice = input("\nEnter choice (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        # Image Segmentation
        image_path = input("Enter image path [input.jpg]: ").strip() or "input.jpg"
        k = int(input("Enter number of clusters k [5]: ").strip() or "5")
        
        try:
            original, segmented, colored, centroids, assignments, history = \
                kmeans_image_segmentation(image_path, k=k)
            
            # Save results
            Image.fromarray(segmented).save(f'segmented_k{k}.png')
            Image.fromarray(colored[:,:,:3]).save(f'colored_k{k}.png')
            print(f"\n✓ Segmented image saved: segmented_k{k}.png")
            print(f"✓ Colored visualization saved: colored_k{k}.png")
            
            # Visualize
            visualize_results(original, segmented, colored, centroids, history, k)
            
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found!")
            print("Creating test image...")
            from PIL import ImageDraw
            img = Image.new('RGB', (800, 600))
            draw = ImageDraw.Draw(img)
            for i in range(5):
                x = np.random.randint(100, 700)
                y = np.random.randint(100, 500)
                color = tuple(np.random.randint(0, 255, 3))
                draw.ellipse([x-50, y-50, x+50, y+50], fill=color)
            img.save('test_input.png')
            print("Test image created: test_input.png")
    
    elif choice == "2":
        # 2D Demo
        demo_2d_clustering()
    
    elif choice == "3":
        # Elbow Method
        print("Creating sample image data...")
        # ใช้ภาพทดสอบ
        from PIL import ImageDraw
        img = Image.new('RGB', (400, 300))
        draw = ImageDraw.Draw(img)
        for i in range(3):
            x = np.random.randint(50, 350)
            y = np.random.randint(50, 250)
            color = tuple(np.random.randint(0, 255, 3))
            draw.ellipse([x-40, y-40, x+40, y+40], fill=color)
        
        img_array = np.array(img)
        data = img_array.reshape(-1, 3).astype(np.float32)
        
        elbow_method(data, max_k=10)
    
    print("\n" + "="*70)
    print("  Done!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Install: pip install pyopencl pillow numpy matplotlib")
        print("2. Check OpenCL: python -c \"import pyopencl as cl; print(cl.get_platforms())\"")
        print("3. Place an image file named 'input.jpg' in the directory")
