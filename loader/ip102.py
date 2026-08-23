import torch, os, json, numpy as np, PIL
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import torchvision.transforms as transforms
from utils.data_loader import get_statistics
import logging

logger = logging.getLogger(__name__)

def find_ip102_root(start_paths: Optional[List[str]] = None) -> str:
    """Auto-discover IP102 dataset root directory."""
    if start_paths is None:
        start_paths = [
            os.environ.get('IP102_DATA_ROOT', ''),
            '/kaggle/input',
            '/content',
            'D:/Sau_Benh_object/retrieval-img/IP102 dataset',
            './IP102 dataset',
            '../IP102 dataset',
            '../../IP102 dataset',
        ]
    
    for base in start_paths:
        if not base:
            continue
        for root, dirs, files in os.walk(base):
            if 'train.json' in files and 'filtered_class.txt' in files:
                jpeg_dir = os.path.join(root, 'VOC2007', 'VOC2007', 'JPEGImages')
                if os.path.exists(jpeg_dir):
                    return root
            if 'JPEGImages' in dirs:
                jpeg_dir = os.path.join(root, 'JPEGImages')
                if os.path.exists(os.path.join(jpeg_dir, 'IP000000000.jpg')) or \
                   any(f.startswith('IP') and f.endswith('.jpg') for f in os.listdir(jpeg_dir)[:10]):
                    parent = os.path.dirname(root)
                    if 'train.json' in os.listdir(parent) or 'filtered_class.txt' in os.listdir(parent):
                        return parent
    raise FileNotFoundError("IP102 dataset not found. Set IP102_DATA_ROOT env var.")

def load_filtered_classes(filtered_class_path: str) -> List[int]:
    """Load the 25 class IDs from filtered_class.txt."""
    with open(filtered_class_path, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

def load_class_names(classes_path: str) -> Dict[int, str]:
    """Load class ID to name mapping from classes.txt."""
    mapping = {}
    with open(classes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                class_id = int(parts[0])
                class_name = parts[1].strip()
                mapping[class_id] = class_name
    return mapping

class IP102(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        prefix: str = "collections/ip102/",
        mode: str = "train",
        session_id: int = 0,
        joint_train: bool = False,
        exp_name: str = "disjoint",
        transform=None,
    ):
        assert mode in ["train", "gallery", "val", "test"], "mode should be {train, gallery, val, test}"
        
        if root is None:
            root = find_ip102_root()
        
        self.root = root
        self.jpeg_dir = os.path.join(root, 'VOC2007', 'VOC2007', 'JPEGImages')
        self.mode = mode
        self.session_id = session_id
        self.joint_train = joint_train
        self.exp_name = exp_name
        
        self.filtered_classes = load_filtered_classes(os.path.join(root, 'filtered_class.txt'))
        self.class_names = load_class_names(os.path.join(root, 'classes.txt'))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.filtered_classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.filtered_classes)}
        
        self.num_classes = len(self.filtered_classes)
        
        collection_prefix = os.path.join(os.path.dirname(__file__), '..', prefix)
        collection_prefix = os.path.normpath(collection_prefix)
        
        if mode in ["train", "gallery"]:
            collection_name = f"ip102_train_{exp_name}_cls{self.num_classes}_task{session_id}.json"
        elif mode == "val":
            collection_name = f"ip102_val_{exp_name}_cls{self.num_classes}_task{session_id}.json"
        else:
            collection_name = f"ip102_test_cls{self.num_classes}_task{session_id}.json"
        
        collection_path = os.path.join(collection_prefix, collection_name)
        
        if not os.path.exists(collection_path):
            self._generate_collections(collection_prefix)
        
        with open(collection_path, 'r') as f:
            datalist = json.load(f)
        
        self.data = np.array([item["file_name"] for item in datalist])
        self.targets = [item["label"] for item in datalist]
        
        if mode == "train":
            self.desc = "training set"
        elif mode == "gallery":
            self.desc = "gallery set"
        elif mode == "val":
            self.desc = "validation query set"
        elif mode == "test":
            self.desc = "testing query set"
        
        mean, std, _, inp_size, _ = get_statistics(dataset='imagenet100')
        if transform is None:
            if mode == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((inp_size, inp_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
        else:
            self.transform = transform
        
        logger.info(f"[{self.desc}] Loaded {len(self.data)} samples, {len(set(self.targets))} classes")

    def _generate_collections(self, collection_prefix: str):
        """Generate collection JSON files from COCO annotations."""
        os.makedirs(collection_prefix, exist_ok=True)
        
        train_json = os.path.join(self.root, 'train.json')
        val_json = os.path.join(self.root, 'val.json')
        test_json = os.path.join(self.root, 'test.json')
        
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        
        train_anns = {ann['image_id']: ann for ann in train_data['annotations']}
        val_anns = {ann['image_id']: ann for ann in val_data['annotations']}
        test_anns = {ann['image_id']: ann for ann in test_data['annotations']}
        
        train_imgs = {img['id']: img['file_name'] for img in train_data['images']}
        val_imgs = {img['id']: img['file_name'] for img in val_data['images']}
        test_imgs = {img['id']: img['file_name'] for img in test_data['images']}
        
        train_items = []
        for img_id, file_name in train_imgs.items():
            if img_id in train_anns:
                cat_id = train_anns[img_id]['category_id']
                if cat_id in self.filtered_classes:
                    train_items.append({
                        "file_name": file_name,
                        "label": self.class_to_idx[cat_id]
                    })
        
        val_items = []
        for img_id, file_name in val_imgs.items():
            if img_id in val_anns:
                cat_id = val_anns[img_id]['category_id']
                if cat_id in self.filtered_classes:
                    val_items.append({
                        "file_name": file_name,
                        "label": self.class_to_idx[cat_id]
                    })
        
        test_items = []
        for img_id, file_name in test_imgs.items():
            if img_id in test_anns:
                cat_id = test_anns[img_id]['category_id']
                if cat_id in self.filtered_classes:
                    test_items.append({
                        "file_name": file_name,
                        "label": self.class_to_idx[cat_id]
                    })
        
        from sklearn.model_selection import train_test_split
        
        train_by_class = {}
        for item in train_items:
            lbl = item['label']
            if lbl not in train_by_class:
                train_by_class[lbl] = []
            train_by_class[lbl].append(item)
        
        val_by_class = {}
        for item in val_items:
            lbl = item['label']
            if lbl not in val_by_class:
                val_by_class[lbl] = []
            val_by_class[lbl].append(item)
        
        test_by_class = {}
        for item in test_items:
            lbl = item['label']
            if lbl not in test_by_class:
                test_by_class[lbl] = []
            test_by_class[lbl].append(item)
        
        # Use contiguous class order for task splits (no shuffle)
        # This ensures global labels are contiguous per task: task 0=0-6, task 1=7-12, etc.
        class_order = list(range(self.num_classes))
        
        task_splits = [7, 6, 6, 6]
        task_classes = []
        start = 0
        for split in task_splits:
            task_classes.append(class_order[start:start+split])
            start += split
        
        for task_id, classes_in_task in enumerate(task_classes):
            train_task = []
            val_task = []
            for cls in classes_in_task:
                train_task.extend(train_by_class.get(cls, []))
                val_task.extend(val_by_class.get(cls, []))
            
            if self.exp_name == "disjoint":
                train_collected = []
                val_collected = []
                for t in range(task_id + 1):
                    for cls in task_classes[t]:
                        train_collected.extend(train_by_class.get(cls, []))
                        val_collected.extend(val_by_class.get(cls, []))
                train_task = train_collected
                val_task = val_collected
            elif "blur" in self.exp_name:
                train_task = train_items
                val_task = val_items
            
            train_collection = f"ip102_train_{self.exp_name}_cls{self.num_classes}_task{task_id}.json"
            val_collection = f"ip102_val_{self.exp_name}_cls{self.num_classes}_task{task_id}.json"
            
            with open(os.path.join(collection_prefix, train_collection), 'w') as f:
                json.dump(train_task, f)
            with open(os.path.join(collection_prefix, val_collection), 'w') as f:
                json.dump(val_task, f)
            
            logger.info(f"Generated {train_collection}: {len(train_task)} samples")
            logger.info(f"Generated {val_collection}: {len(val_task)} samples")
        
        test_collections = []
        for task_id in range(len(task_splits)):
            if self.exp_name == "disjoint":
                test_task = []
                for t in range(task_id + 1):
                    for cls in task_classes[t]:
                        test_task.extend(test_by_class.get(cls, []))
            elif "blur" in self.exp_name:
                test_task = test_items
            else:
                test_task = test_items
            
            test_collection = f"ip102_test_cls{self.num_classes}_task{task_id}.json"
            with open(os.path.join(collection_prefix, test_collection), 'w') as f:
                json.dump(test_task, f)
            test_collections.append(test_collection)
            logger.info(f"Generated {test_collection}: {len(test_task)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data[idx]
        label = self.targets[idx]
        img_path = os.path.join(self.jpeg_dir, img_name)
        imgpil = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            trans_img = self.transform(imgpil)
        if self.mode == "gallery":
            return trans_img, label, img_name
        else:
            return trans_img, label

    def add_memory(self, exem_data, exem_labels):
        exem_data = exem_data.tolist()
        exem_labels = exem_labels.tolist()
        tmp = self.data.tolist()
        tmp.extend(exem_data)
        self.data = np.array(tmp)
        self.targets.extend(exem_labels)

    def show(self, verbose=True):
        print("-----------------")
        print(f"[{self.desc}]")
        print(f"class label from {np.min(self.targets)} to {np.max(self.targets)}")
        print(f"number of data: {self.data.shape} with dtype {self.data.dtype}")
        if verbose:
            unique, counts = np.unique(self.targets, return_counts=True)
            print({int(tar): int(cnt) for tar, cnt in zip(unique, counts)})
        print("-----------------")

    def get_class_name(self, idx: int) -> str:
        """Get original class name for a given index (0-24)."""
        original_id = self.idx_to_class[idx]
        return self.class_names.get(original_id, f"class_{original_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = IP102(mode="train", session_id=0, exp_name="disjoint")
    ds.show()
    print("Class names:", [ds.get_class_name(i) for i in range(ds.num_classes)])