from PIL import Image
import numpy as np

class ObjectDetector:
    def __init__(self, image_path, dbscan_eps = None, dbscan_min_samples = 5, max_iterations = 200):
        self.image = np.array(Image.open(image_path).convert('RGB'))
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        
        self.dbscan_eps = dbscan_eps if dbscan_eps is not None else min(self.height, self.width) / 100
        self.dbscan_min_samples = dbscan_min_samples
        self.max_iterations = max_iterations

    def detect_objects(self):
        grayscale_image = self.to_grayscale()
        blurred_image = self.blur(grayscale_image, kernel_size = 3)
        edges = self.sobel_edge_detection(blurred_image)
        binary_image = self.threshold(edges, threshold_value = 15)

        data_points = np.argwhere(binary_image == 1)

        if data_points.size == 0:
            print("No foreground pixels detected.")
            return []
        
        clusters = self.dbscan(data_points)

        unique_clusters = set(clusters)
        unique_clusters.discard(-1)
        
        objects = []
        for cluster_id in unique_clusters:
            cluster_points = data_points[clusters == cluster_id]
            if cluster_points.size == 0:
                continue

            y_min, x_min = cluster_points.min(axis = 0)
            y_max, x_max = cluster_points.max(axis = 0)

            center = ((y_min + y_max) // 2, (x_min + x_max) // 2)
            radius = max(x_max - x_min, y_max - y_min) // 2

            objects.append({
                'center': tuple(center),
                'radius': int(radius),
            })

        return objects

    def to_grayscale(self):
        r = self.image[:, :, 0].astype(np.float64)
        g = self.image[:, :, 1].astype(np.float64)
        b = self.image[:, :, 2].astype(np.float64)

        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def blur(self, image, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) 
        kernel /= kernel_size ** 2
        return self.convolve(image, kernel)

    def sobel_edge_detection(self, image):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float64)

        grad_x = self.convolve(image, sobel_x)
        grad_y = self.convolve(image, sobel_y)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        max_val = magnitude.max()
        
        if max_val > 0:
            magnitude = (magnitude / max_val) * 255.0

        return magnitude.astype(np.uint8)

    def convolve(self, image, kernel):
        kernel_height, kernel_width = kernel.shape
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode = 'constant', constant_values = 0)

        convolved = np.zeros_like(image, dtype=np.float64)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i : i + kernel_height, j : j + kernel_width]
                convolved[i, j] = np.sum(region * kernel)

        return convolved

    def threshold(self, image, threshold_value = 25, border_size = 5):
        thresholded = (image > threshold_value).astype(np.uint8)

        thresholded[:border_size, :] = 0
        thresholded[-border_size:, :] = 0
        thresholded[:, :border_size] = 0
        thresholded[:, -border_size:] = 0

        return thresholded
        
    def dbscan(self, points): 
        clusters = [0] * len(points)
        cur_id = 0
        
        for i in range(len(points)):
            if clusters[i] != 0:
                continue
            
            neighbors = self.region_query(points, i)
            
            if len(neighbors) < self.dbscan_min_samples:
                clusters[i] = -1
            else: 
                cur_id += 1
                self.grow_cluster(points, clusters, i, neighbors, cur_id)
        
        return np.array(clusters)

    def grow_cluster(self, points, clusters, start_point_i, neighbor_points, cur_cluster_id):
        clusters[start_point_i] = cur_cluster_id
        i = 0
        
        while i < len(neighbor_points):
            p = neighbor_points[i]
            
            if clusters[p] == 0:
                clusters[p] = cur_cluster_id
                p_neighbors = self.region_query(points, p)
                
                if len(p_neighbors) >= self.dbscan_min_samples:
                    neighbor_points.extend(p_neighbors)
            
            elif clusters[p] == -1:
                clusters[p] = cur_cluster_id
            
            i += 1

    def region_query(self, points, start_point):
        distances = np.linalg.norm(points - points[start_point], axis=1)
        neighbors = np.where(distances < self.dbscan_eps)[0]
        return neighbors.tolist()
