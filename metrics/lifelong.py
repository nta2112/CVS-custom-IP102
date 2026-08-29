import numpy as np
import json
import os

class LifelongMetric:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.mAP_matrix = []

    @staticmethod
    def _clean_scores(scores):
        arr = np.asarray(scores, dtype=float)
        arr = arr[np.isfinite(arr)]
        return arr
    
    def update(self, task_id, map_scores_per_class):
        """Update mAP matrix with scores for current task.
        
        Args:
            task_id: current task index (0-based)
            map_scores_per_class: list of mAP scores for each class in current task
        """
        while len(self.mAP_matrix) <= task_id:
            self.mAP_matrix.append([])
        self.mAP_matrix[task_id] = map_scores_per_class
    
    def compute_plasticity(self):
        """Plasticity: average mAP on current task classes after learning current task."""
        if not self.mAP_matrix:
            return 0.0
        last_task = self._clean_scores(self.mAP_matrix[-1])
        return float(np.mean(last_task)) if last_task.size else 0.0
    
    def compute_forgetting(self):
        """Forgetting: average drop in mAP on previous task classes."""
        if len(self.mAP_matrix) < 2:
            return 0.0
        
        total_forgetting = 0.0
        count = 0
        
        for prev_task in range(len(self.mAP_matrix) - 1):
            prev_scores = self._clean_scores(self.mAP_matrix[prev_task])
            curr_scores = self._clean_scores(self.mAP_matrix[-1][:min(len(self.mAP_matrix[-1]), len(self.mAP_matrix[prev_task]))])
            
            if prev_scores.size > 0 and curr_scores.size > 0:
                forgetting = np.mean(prev_scores) - np.mean(curr_scores)
                total_forgetting += max(0.0, float(forgetting))
                count += 1
        
        return total_forgetting / count if count > 0 else 0.0
    
    def compute_overall(self):
        """Overall: average mAP across all seen classes after final task."""
        if not self.mAP_matrix:
            return 0.0
        
        all_scores = []
        for task_scores in self.mAP_matrix:
            all_scores.extend(self._clean_scores(task_scores).tolist())
        
        return float(np.mean(all_scores)) if all_scores else 0.0
    
    def get_results(self):
        return {
            'Plasticity': self.compute_plasticity(),
            'Forgetting': self.compute_forgetting(),
            'Overall': self.compute_overall(),
        }


def save_results_csv(save_dir, task_id, num_classes, cnn_top1, nme_top1, 
                     retrieval_metrics, openworld_metrics, lifelong_metrics):
    """Save results to CSV file."""
    csv_path = os.path.join(save_dir, 'results.csv')

    def clean_value(value):
        if value is None:
            return None
        try:
            v = float(value)
            if np.isfinite(v):
                return v
            return None
        except (TypeError, ValueError):
            return value
    
    row = {
        'task': task_id,
        'numclass': num_classes,
        'cnn_top1': clean_value(cnn_top1),
        'nme_top1': clean_value(nme_top1),
        'R@1': clean_value(retrieval_metrics.get('R@1', 0)),
        'R@5': clean_value(retrieval_metrics.get('R@5', 0)),
        'R@10': clean_value(retrieval_metrics.get('R@10', 0)),
        'mAP': clean_value(retrieval_metrics.get('mAP_macro', 0)),
        'R@1_S': clean_value(openworld_metrics.get('R@1_S', None)),
        'R@1_U': clean_value(openworld_metrics.get('R@1_U', None)),
        'AUROC': clean_value(openworld_metrics.get('AUROC', None)),
        'FPR95': clean_value(openworld_metrics.get('FPR@TPR95', None)),
        'Plasticity': clean_value(lifelong_metrics.get('Plasticity', 0)),
        'Forgetting': clean_value(lifelong_metrics.get('Forgetting', 0)),
        'Overall': clean_value(lifelong_metrics.get('Overall', 0)),
    }
    
    header = ['task', 'numclass', 'cnn_top1', 'nme_top1', 'R@1', 'R@5', 'R@10', 'mAP',
              'R@1_S', 'R@1_U', 'AUROC', 'FPR95', 'Plasticity', 'Forgetting', 'Overall']
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(str(row[h]) if row[h] is not None else '' for h in header) + '\n')


def save_history_json(save_dir, history):
    """Save history to JSON file."""
    json_path = os.path.join(save_dir, 'history.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def load_history_json(save_dir):
    """Load history from JSON file."""
    json_path = os.path.join(save_dir, 'history.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    lm = LifelongMetric(4)
    lm.update(0, [0.8, 0.7, 0.9, 0.85, 0.75, 0.8, 0.9])
    print("After task 0:", lm.get_results())
    lm.update(1, [0.75, 0.7, 0.85, 0.8, 0.7, 0.75])
    print("After task 1:", lm.get_results())
    lm.update(2, [0.7, 0.65, 0.8, 0.75, 0.65, 0.7])
    print("After task 2:", lm.get_results())
    lm.update(3, [0.65, 0.6, 0.75, 0.7, 0.6, 0.65])
    print("After task 3:", lm.get_results())