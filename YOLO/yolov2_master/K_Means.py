import numpy as np 


class K_Means:
    
    
    def __init__(self, k, boxes):
        
        self.k = k
        self.boxes = boxes
        self.rows = self.boxes.shape[0]
        self.distances = np.empty((self.rows, self.k))
        self.last_centroids = np.zeros((self.rows,))
        
        self.boxes = self.process_boxes(self.boxes)
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(self.boxes[i,:])
        
        self.centroids = np.asarray(self.centroids, dtype=np.float32)
        
    def process_boxes(self, boxes):
        '''
        This assumes that the boxes are normalized to their width and height, therefore all boxes will be overlapping each other.
        We don't care about the x,y coordinates since we only want the height and width for the centroid boxes.
        '''
        new_boxes = boxes.copy()
        for row in range(self.rows):
            
            new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
            new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
        
        return np.delete(new_boxes, [0,1], axis=1)
    
    def iou(self, box, centroids):
        
        x = np.minimum(centroids[:, 0], box[0])
        y = np.minimum(centroids[:, 1], box[1])
        
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("The given box has no area!")
        
        intersection_area = x * y
        box_area = box[0] * box[1]
        centroid_area = centroids[:, 0] * centroids[:, 1]
        
        IoUs = intersection_area / (box_area + centroid_area - intersection_area)
        
        return IoUs
    
    def __call__(self):
        
        
        while True:
            
            for row in range(self.rows):
                self.distances[row] = 1 - self.iou(self.boxes[row], self.centroids)
            
            nearest_centroids = np.argmin(self.distances, axis=1)
            
            if (self.last_centroids == nearest_centroids).all():
                break
            
            for cluster in range(self.k):
                self.centroids[cluster] = np.mean(self.boxes[nearest_centroids == cluster], axis=0)
                
            self.last_centroids = nearest_centroids
        
        return self.centroids