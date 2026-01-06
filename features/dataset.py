import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, root_dataset, image_size=(224, 224)):
        self.image_paths = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        class_names = sorted([
            d for d in os.listdir(root_dataset)
            if os.path.isdir(os.path.join(root_dataset, d))
        ], key=lambda x: int(x))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dataset, class_name)
            label = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
def create_dataset(root_dataset, input_size):
    return TrainDataset(root_dataset, (input_size[1], input_size[2]))

def create_dataloader(dataset, bs, shuffle):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle)