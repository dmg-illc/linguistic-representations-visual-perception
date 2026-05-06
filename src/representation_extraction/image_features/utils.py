from torch.utils.data import Dataset

import os 
from PIL import Image

class NSDDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform

        # Filter to include only image files
        self.image_files = [f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img_id = self.image_files[idx].split('nsd')[-1].split('.png')[0].lstrip('0')
        image = Image.open(img_path).convert('RGB')  


        image = self.transform(image)
        return img_id, image