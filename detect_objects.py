from collections import deque
from PIL import Image
import numpy as np

class ObjectDetector:
    def __init__(self, image_path, dbscan_eps = None, dbscan_min_samples = 5, max_iterations = 20):
        self.image = np.array(Image.open(image_path).convert('RGB'))
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        
        self.dbscan_eps = dbscan_eps if dbscan_eps is not None else min(self.height, self.width) / 100 # we may need to change this. For bigger balls, decrease 100, for smaller, increase
        self.dbscan_min_samples = dbscan_min_samples
        self.max_iterations = max_iterations

    def detect_objects(self):
        grayscale_image = self.to_grayscale()
        blurred_image = self.blur(grayscale_image, kernel_size = 3)
        edges = self.sobel_edge_detection(blurred_image)
        binary_image = self.threshold(edges, threshold_value = 15)

        data_points = np.argwhere(binary_image == 1)

        if data_points.size == 0:
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
        
        # DBSCAN deals with touching edges, but it doesn't work (if epsilon is not big enough) when one object is inside another, so we need to remove inner balls and only leave outer ones.
        filtered_objects = []
        for i, obj1 in enumerate(objects):
            is_inside = False
            
            for j, obj2 in enumerate(objects):
                if i != j:
                    distance = np.linalg.norm(np.array(obj1['center']) - np.array(obj2['center']))
                    if distance + obj1['radius'] <= obj2['radius']:
                        is_inside = True
                        break
                    
            if not is_inside:
                filtered_objects.append(obj1)

        return filtered_objects

    def to_grayscale(self):
        r = self.image[:, :, 0].astype(np.float64)
        g = self.image[:, :, 1].astype(np.float64)
        b = self.image[:, :, 2].astype(np.float64)

        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def blur(self, image, kernel_size = 3):
        kernel = np.ones((kernel_size, kernel_size), dtype = np.float64)
        kernel /= (kernel_size ** 2)
        return self.convolve(image, kernel)

    def sobel_edge_detection(self, image):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype = np.float64)
        sobel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype = np.float64)

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

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                              mode = 'constant', constant_values = 0)

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
        #  0  => unvisited
        # -1  => noise
        # >0  => cluster IDs
        clusters = np.zeros(len(points), dtype = int)
        cluster_id = 0

        for i in range(len(points)):
            if clusters[i] != 0:
                continue
            
            neighbors = self.region_query(points, i)

            if len(neighbors) < self.dbscan_min_samples:
                clusters[i] = -1
            else:
                cluster_id += 1
                self.grow_cluster(points, clusters, i, neighbors, cluster_id)

        return clusters

    def grow_cluster(self, points, clusters, start_idx, neighbors, cluster_id):
        queue = deque(neighbors)
        clusters[start_idx] = cluster_id

        while queue:
            current_idx = queue.popleft()

            if clusters[current_idx] == 0 or clusters[current_idx] == -1:
                clusters[current_idx] = cluster_id
                current_neighbors = self.region_query(points, current_idx)
                
                if len(current_neighbors) >= self.dbscan_min_samples:
                    for idx in current_neighbors:
                        if clusters[idx] == 0 or clusters[idx] == -1:
                            queue.append(idx)

    def region_query(self, points, idx):
        distances = np.linalg.norm(points - points[idx], axis = 1)
        return np.where(distances < self.dbscan_eps)[0]
