import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import os
import pandas as pd
from PIL import Image
import polars as pl

class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet18, self).__init__()
        
        # Load pretrained ResNet18 model
        self.model = models.resnet18() # weights='DEFAULT'
        
        # Modify first conv layer to accept 1-channel grayscale instead of 3-channel RGB
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify classifier to match number of disease classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class FineTunedResNet50(nn.Module):
    """Same idea as resnet 18, just bigger"""
    def __init__(self, num_classes):
        super(FineTunedResNet50, self).__init__()
        
        self.model = models.resnet50(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class FineTunedDenseNet121(nn.Module):
    """Same idea as the resnets, just a densenet now"""
    def __init__(self, num_classes):
        super(FineTunedDenseNet121, self).__init__()
        
        self.model = models.densenet121(weights='DEFAULT')
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class SimpleXrayCNN(nn.Module):
    """Pretty much a resnet model"""
    def __init__(self, num_classes):
        super(SimpleXrayCNN, self).__init__()
        
        # Input: 1 x 1024 x 1024
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),  # Output: 16 x 512 x 512
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 16 x 256 x 256
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # Output: 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 32 x 64 x 64
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16
            
            # Fourth convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 8 x 8
            
            # Fifth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 4 x 4

        )
        
        # Global average pooling to handle variable image sizes
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Check initial input shape
        x = self.features(x)
        # print(f"After feature shape: {x.shape}")  # (batch, 256, 4, 4)
        # x = self.global_pool(x)
        x = self.classifier(x)
        return x

def setup_model_and_training(dataset_xrays, batch_size=16, learning_rate=0.001, model_type="simpleCNN", model_mode="single"):
    """Sets up a model, loads data into a dataloader and defines the modeltype and the model mode according to wishes
    
    If using the multi attention model, this should be """
    # Create data loader
    if model_type=="MultiAttention":
        data_loader = XrayMultiLabelDataset(os.path.join("..", "data", "images_train"))
    else:
        data_loader = DataLoader(dataset_xrays, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Get number of classes
    num_classes = len(data_loader.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {data_loader.classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "simpleCNN":
        model = SimpleXrayCNN(num_classes=num_classes)
    elif model_type == "ResNet18":
        model = FineTunedResNet18(num_classes=num_classes)
    elif model_type == "ResNet50":
        model = FineTunedResNet50(num_classes=num_classes)
    elif model_type == "DenseNet121":
        model = FineTunedDenseNet121(num_classes=num_classes)
    elif model_type == "MultiAttention":
        model = MultiAttentionXrayCNN(num_classes=num_classes)
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    if model_mode == "single":
        criterion = nn.CrossEntropyLoss()
    elif model_mode == "multi":
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, data_loader, criterion, optimizer, device

def setup_dataloader(dataset_xrays, batch_size=16, num_workers=4, shuffle=True):
    """Create a dataloader"""
    data_loader = DataLoader(dataset_xrays, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return data_loader


def train_model(model, data_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Training epoch number: {epoch}")
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            # print(f"Shape of inputs {inputs.shape}")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    
    return model

def train_multilabel_model(model, data_loader, criterion, optimizer, device, num_epochs=10):
    """Should be combined with original function"""
    model.train()
    
    # Use Binary Cross Entropy loss for multi-label classification
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Build class mapping (finding name to index)
    class_mapping = {class_name: idx for idx, class_name in enumerate(data_loader.classes)} # Which dataset? 
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels, filenames in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with filenames and class mapping: What is this image according to the model?
            # output is 15 probabilities of each finding
            outputs = model(inputs, filenames, class_mapping)
            
            # Compute loss: did we classify this image correctly?
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(data_loader)
        print(f"Data loader length: {len(data_loader)}")
        # epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model

class MultiAttentionXrayCNN(nn.Module):
    def __init__(self, num_classes, bbox_file_path=os.path.join("..", "data", "classification", "BBox_List_2017.csv")):
        super(MultiAttentionXrayCNN, self).__init__()
        
        # Base CNN backbone (same as before)
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),  # Output: 16 x 512 x 512
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 16 x 256 x 256
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # Output: 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 32 x 64 x 64
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16
            
            # Fourth convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 8 x 8
            
            # Fifth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 4 x 4
        )
        
        # Finding-specific attention modules (one per class)
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
        
        # Classifier with multi-label output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Load and structure bounding box data
        self.image_finding_map = {}
        if bbox_file_path:
            self.process_bbox_data()
    

    def process_bbox_data(self, bbox_csv_path=os.path.join("..", "data", "classification", "BBox_List_2017.csv"),
                            data_entry_path=os.path.join("..", "data", "classification", "Data_Entry_2017_v2020.csv"),
                            folder_with_cutoff_images=os.path.join("..", "data", "images_train")):
        """Create a hashmap with bounding box data"""
        bbox_data = pl.read_csv(bbox_csv_path, new_columns=["Image Index", "Finding Label", "x", "y", "w", "h"])
        #Only those actually present should be included. If we dont do this here and in the class dataset definition we get different outputs
        bbox_data = bbox_data.filter(pl.col("Image Index").is_in(os.listdir(folder_with_cutoff_images)))
        bbox_data = bbox_data.with_columns(pl.col("Image Index").str.replace("Infiltrate", "Infiltration"), literal=True)

        data_entry_data = pl.read_csv(data_entry_path)
        data_entry_data = data_entry_data.filter(pl.col("Image Index").is_in(os.listdir(folder_with_cutoff_images)))

        image_finding_map = {}
        
        # Get the bounding box data and insert it
        for row in bbox_data.iter_rows(named=True):
            image_index = row["Image Index"]
            finding = row["Finding Label"]
            bbox = (row["x"], row["y"], row["w"], row["h"])
            
            if image_index not in image_finding_map:
                image_finding_map[image_index] = {}
            
            if finding not in image_finding_map[image_index]:
                image_finding_map[image_index][finding] = []
            
            image_finding_map[image_index][finding].append(bbox)    
    
        # and now for those without bounding boxes.
        for row in data_entry_data.iter_rows(named=True): # for _, row in self.bbox_df.iterrows():
            img_id = row['Image Index']
            findings = [f.strip() for f in row['Finding Labels'].split('|')]
            
            if img_id not in image_finding_map:
                image_finding_map[img_id] = {}
            
            for finding in findings:
                if finding not in image_finding_map[img_id]:
                    image_finding_map[img_id][finding] = []
        
        return image_finding_map

    def create_attention_mask(self, feature_map_size, bboxes, device):
        """Create Gaussian attention mask from multiple bounding boxes"""
        height, width = feature_map_size
        mask = torch.zeros((height, width), device=device)
        
        # Original image dimensions (assuming 1024x1024)
        orig_h, orig_w = 1024, 1024
        
        # Scale factors
        scale_h = height / orig_h
        scale_w = width / orig_w
        
        # Y and X coordinates for the feature map
        y_indices = torch.arange(0, height, device=device)
        x_indices = torch.arange(0, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_indices, x_indices)
        
        # Create combined mask from all bounding boxes for this finding
        for x, y, w, h in bboxes:
            # Scale bbox coordinates
            x_scaled = int(x * scale_w)
            y_scaled = int(y * scale_h)
            w_scaled = max(1, int(w * scale_w))
            h_scaled = max(1, int(h * scale_h))
            
            # Calculate center of bounding box
            center_y = y_scaled + h_scaled // 2
            center_x = x_scaled + w_scaled // 2
            
            # Sigma based on box size
            sigma = max(h_scaled, w_scaled) / 3
            
            # Gaussian mask for this bbox
            bbox_mask = torch.exp(-((y_grid - center_y)**2 + (x_grid - center_x)**2) / (2 * sigma**2))
            
            # Combine with existing mask (take maximum value at each point)
            mask = torch.maximum(mask, bbox_mask)
        
        return mask.unsqueeze(0)  # Add channel dimension
    
    def forward(self, x, image_filenames=None, class_mapping=None):
        """Forward pass of the model"""
        # batch_size = x.shape[0]
        device = x.device
        
        # Extract features through CNN backbone
        features = self.features(x)
        _, _, height, width = features.shape
        
        # Initialize combined features
        # combined_features = torch.zeros_like(features)
        # Initialize combined features with a clone of features to avoid modifying the original tensor
        combined_features = features.clone()


        # Process each image with its specific findings
        for i, filename in enumerate(image_filenames):
            # Get all class-specific attentions
            all_attentions = []
            
            # If we have bbox data for this image
            if filename in self.image_finding_map:
                # For each class/finding that appears in this image
                for finding, bboxes in self.image_finding_map[filename].items():
                    # Get class index from class name
                    if class_mapping and bboxes and finding in class_mapping: # Check if bboxes even is filled or just empty.
                        class_idx = class_mapping[finding]
                        
                        # Create attention mask from bounding boxes for this finding
                        bbox_attention = self.create_attention_mask((height, width), bboxes, device)
                        
                        # Get learned attention for this class
                        learned_attention = self.attention_modules[class_idx](features[i:i+1])
                        
                        # Combine bbox and learned attention
                        combined_attention = bbox_attention * learned_attention
                        
                        # Apply attention to features for this class
                        attended_features = features[i:i+1] * combined_attention
                        
                        # Add to the collection
                        all_attentions.append((class_idx, attended_features))
                
                # If we have attentions for this image
                if all_attentions:
                    # Average or max-pool the attended features across all findings
                    # (design choice: you could use different aggregation methods)
                    # combined_features[i] = torch.cat([attn[1] for attn in all_attentions]).mean(dim=0)
                    combined_features[i] = torch.mean(torch.stack(all_attentions), dim=0)
    
            #     else:
            #         # File name found but no bboxes provided, use original features
            #         combined_features[i] = features[i]
            # else:
            #     # No bounding boxes for this image, use original features
            #     combined_features[i] = features[i]
        
        # If no bounding box data was used at all, just use the original features. No modifications made.
        if torch.all(combined_features == features):
            combined_features = features.clone()
        
        # Classify the attended features
        outputs = self.classifier(combined_features)
        
        return outputs


class XrayMultiLabelDataset(Dataset):
    def __init__(self, root_dir, bbox_csv_path=os.path.join("..", "data", "classification", "BBox_List_2017.csv"), data_entry_path=os.path.join("..", "data", "classification", "Data_Entry_2017_v2020.csv"), transform=None):
        """
        An extension of the standard dataset approach that handles multiple labels per image
        and incorporates bounding box information.
        
        Args:
            root_dir: Directory containing all images
            bbox_csv_path: Path to CSV with format (Image Index, Finding Label, x, y, w, h)
            transform: Image transformations

            TODO: class findings based on the folders that an image is present in. Either from the CSV with findings of from the sort.
            Process this in label tensor.
        """
    
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()  # Converts PIL image to tensor
        ])
        
        # Load bounding box data and data entry data
        bbox_data = pl.read_csv(bbox_csv_path, new_columns=["Image Index", "Finding Label", "x", "y", "w", "h"]) 
        # rename Infiltrate to Infiltration so that it matches the findings data.
        bbox_data = bbox_data.with_columns(pl.col("Image Index").str.replace("Infiltrate", "Infiltration"), literal=True)


        data_entry_data = pl.read_csv(data_entry_path)
        data_entry_data = data_entry_data.filter(pl.col("Image Index").is_in(os.listdir(root_dir)))

        #Only those actually present should be included.
        bbox_data = bbox_data.filter(pl.col("Image Index").is_in(os.listdir(root_dir)))

        # Get all unique image filenames
        # self.image_filenames = bbox_data.select("Image Index").unique()["Image Index"].to_list() 
        self.image_filenames = os.listdir(root_dir)

        # Get all unique finding labels (classes)
        self.classes =  data_entry_data.get_column('Finding Labels').str.split('|').map_elements(lambda x: [f.strip() for f in x], return_dtype=pl.List(pl.Utf8)).explode().unique() # sorted(bbox_data_filtered.select("Finding Label").unique()["Finding Label"].to_list()) #TODO: adjust this variable too.
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build a mapping from image to findings and bboxes
        self.image_findings = {}
        for row in bbox_data.iter_rows(named=True): # for _, row in self.bbox_df.iterrows():
            img_id = row['Image Index']
            finding = row['Finding Label']
            bbox = (row['x'], row['y'], row['w'], row['h'])
            
            if img_id not in self.image_findings:
                self.image_findings[img_id] = {}
                
            if finding not in self.image_findings[img_id]:
                self.image_findings[img_id][finding] = []
                
            # We add it, but its not used in the context of this class
            self.image_findings[img_id][finding].append(bbox)

        # and now for those without bounding boxes.
        for row in data_entry_data.iter_rows(named=True): # for _, row in self.bbox_df.iterrows():
            img_id = row['Image Index']
            findings = [f.strip() for f in row['Finding Labels'].split('|')]
            
            if img_id not in self.image_findings:
                self.image_findings[img_id] = {}
            
            for finding in findings:
                if finding not in self.image_findings[img_id]:
                    self.image_findings[img_id][finding] = []
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get image filename
        img_file = self.image_filenames[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_file)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        
            # Onconditional transformation to Torch tensor, no: if self.transform:
            image = self.transform(image)  # Shape: (H, W) → (1, H, W)

            #Add a layer if needed, since we already have a greyscale pic.
            if image.ndimension() == 3:  # Should be (1, H, W) already
                image = image.unsqueeze(0)  # Add batch dimension → (1, 1, H, W)
        else:
            image = None

        # Create multi-hot encoding for labels
        # This needs  to be done with data entry
        label_tensor = torch.zeros(len(self.classes))
        if img_file in self.image_findings:
            for finding, _ in self.image_findings[img_file].items(): #BBoxes not used 
                label_tensor[self.class_to_idx[finding]] = 1
        label_tensor = label_tensor.unsqueeze(0)

        return image, label_tensor, img_file
    

"""General functions begin here"""
def save_model(model, save_path='xray_model.pth', save_metadata=True):
    """
    Save the trained model and optionally its metadata.
    
    Args:
        model (nn.Module): The trained PyTorch model
        save_path (str): Path to save the model
        save_metadata (bool): Whether to save model metadata
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model state dictionary
    torch.save(model.state_dict(), save_path)
    
    # Save model architecture and metadata
    if save_metadata:
        metadata_path = os.path.splitext(save_path)[0] + '_metadata.pth'
        
        # Store additional information
        metadata = {
            'architecture': str(model),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'timestamp': time.strftime("%Y-%m-%d-%H:%M:%S"),
        }
        
        torch.save(metadata, metadata_path)
        print(f"Model saved to {save_path}")
        print(f"Model metadata saved to {metadata_path}")
        print(f"Total parameters: {metadata['num_parameters']:,}")
        print(f"Trainable parameters: {metadata['trainable_parameters']:,}")


def load_model(model_class, model_path, num_classes, device=None):
    """
    Load a saved model.
    
    Args:
        model_class: The model class to instantiate
        model_path (str): Path to the saved model
        num_classes (int): Number of classes for model initialization
        device (torch.device): Device to load the model to
        
    Returns:
        nn.Module: The loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Try to load metadata
    metadata_path = os.path.splitext(model_path)[0] + '_metadata.pth'
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location=device)
        print(f"Model metadata loaded from {metadata_path}")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    
    return model
