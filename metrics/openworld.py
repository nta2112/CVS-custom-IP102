import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

class OpenWorldMetric:
    def __init__(self):
        self.requires = ['nearest_features_cosine']
    
    def __call__(self, query_labels, k_closest_classes, seen_classes, all_seen=False):
        """
        query_labels: (N, 1) array of ground truth labels
        k_closest_classes: (N, max_k) array of predicted class labels for top-k neighbors
        seen_classes: set/list of class indices that have been seen so far
        all_seen: bool, if True, return None for OOD metrics (as per spec)
        """
        query_labels = query_labels.flatten()
        seen_classes = set(seen_classes)
        
        results = {}
        
        is_seen = np.array([lbl in seen_classes for lbl in query_labels])
        is_unseen = ~is_seen
        
        if all_seen or not is_unseen.any():
            results['R@1_S'] = self.compute_recall_at_1(query_labels[is_seen], k_closest_classes[is_seen]) if is_seen.any() else None
            results['R@1_U'] = None
            results['AUROC'] = None
            results['FPR@TPR95'] = None
            return results
        
        results['R@1_S'] = self.compute_recall_at_1(query_labels[is_seen], k_closest_classes[is_seen]) if is_seen.any() else 0.0
        results['R@1_U'] = self.compute_recall_at_1(query_labels[is_unseen], k_closest_classes[is_unseen]) if is_unseen.any() else 0.0
        
        ood_scores = self.compute_ood_scores(k_closest_classes, seen_classes)
        results['AUROC'] = self.compute_auroc(is_unseen, ood_scores)
        results['FPR@TPR95'] = self.compute_fpr_at_tpr95(is_unseen, ood_scores)
        
        return results
    
    def compute_recall_at_1(self, query_labels, k_closest_classes):
        if len(query_labels) == 0:
            return 0.0
        correct = sum(1 for i in range(len(query_labels)) if query_labels[i] == k_closest_classes[i, 0])
        return correct / len(query_labels)
    
    def compute_ood_scores(self, k_closest_classes, seen_classes):
        """Compute OOD score as 1 - max similarity to seen classes."""
        scores = []
        for i in range(len(k_closest_classes)):
            top1 = k_closest_classes[i, 0]
            if top1 in seen_classes:
                scores.append(0.0)
            else:
                scores.append(1.0)
        return np.array(scores)
    
    def compute_auroc(self, is_unseen, ood_scores):
        try:
            return roc_auc_score(is_unseen, ood_scores)
        except ValueError:
            return 0.5
    
    def compute_fpr_at_tpr95(self, is_unseen, ood_scores):
        try:
            fpr, tpr, thresholds = roc_curve(is_unseen, ood_scores)
            idx = np.argmin(np.abs(tpr - 0.95))
            return fpr[idx]
        except ValueError:
            return 1.0


def compute_openworld_metrics(query_labels, k_closest_classes, seen_classes, all_seen=False):
    metric = OpenWorldMetric()
    return metric(query_labels, k_closest_classes, seen_classes, all_seen)


if __name__ == "__main__":
    query_labels = np.array([[0], [1], [2], [3], [4]])
    k_closest = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [5, 6, 7],
        [5, 6, 7],
    ])
    seen = {0, 1, 2}
    print(compute_openworld_metrics(query_labels, k_closest, seen))
    print(compute_openworld_metrics(query_labels, k_closest, {0,1,2,3,4}, all_seen=True))