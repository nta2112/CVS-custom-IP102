import numpy as np
from sklearn.metrics import average_precision_score

class RetrievalMetric:
    def __init__(self, ks=(1, 5, 10)):
        self.ks = ks
        self.requires = ['nearest_features_cosine']
    
    def __call__(self, query_labels, k_closest_classes):
        """
        query_labels: (N, 1) array of ground truth labels
        k_closest_classes: (N, max_k) array of predicted class labels for top-k neighbors
        """
        query_labels = query_labels.flatten()
        results = {}
        
        for k in self.ks:
            correct = 0
            for i in range(len(query_labels)):
                if query_labels[i] in k_closest_classes[i, :k]:
                    correct += 1
            results[f'R@{k}'] = correct / len(query_labels)
        
        map_macro, per_class_map = self.compute_map_macro(query_labels, k_closest_classes)
        results['mAP_macro'] = map_macro
        results['per_class_map'] = per_class_map
        return results
    
    def compute_map_macro(self, query_labels, k_closest_classes):
        """Compute macro-averaged mAP across all classes and per-class mAP."""
        unique_classes = np.unique(query_labels)
        aps = []
        per_class_map = {}
        
        for cls in unique_classes:
            cls_mask = (query_labels == cls)
            if not cls_mask.any():
                continue
            
            cls_queries = query_labels[cls_mask]
            cls_predictions = k_closest_classes[cls_mask]
            
            y_true = []
            y_scores = []
            
            for i in range(len(cls_queries)):
                for rank, pred in enumerate(cls_predictions[i]):
                    y_true.append(1 if pred == cls else 0)
                    y_scores.append(1.0 / (rank + 1))
            
            if sum(y_true) > 0:
                ap = average_precision_score(y_true, y_scores)
                aps.append(ap)
                per_class_map[int(cls)] = ap
            else:
                per_class_map[int(cls)] = 0.0
        
        return (np.mean(aps) if aps else 0.0), per_class_map


def compute_retrieval_metrics(query_labels, k_closest_classes, ks=(1, 5, 10)):
    metric = RetrievalMetric(ks)
    return metric(query_labels, k_closest_classes)


if __name__ == "__main__":
    query_labels = np.array([[0], [1], [2], [0], [1]])
    k_closest = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [0, 2, 1],
        [1, 2, 0],
    ])
    print(compute_retrieval_metrics(query_labels, k_closest))