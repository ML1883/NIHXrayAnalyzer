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
from torch.utils.tensorboard import SummaryWriter
from io import StringIO
import json
from sklearn.metrics import accuracy_score, f1_score

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
            # nn.Dropout(0.2),
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # Output: 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Output: 32 x 64 x 64
            # nn.Dropout(0.2),

            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16
            # nn.Dropout(0.2),

            # Fourth convolutional block
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),  # New output: 256 X 16 X 16  Old Output: 128 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),  # New output.: 256 X 4 X 4 Old Output: 128 x 8 x 8
            # nn.Dropout(0.2),
            
            # Fifth convolutional block
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 256 x 8 x 8
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 4 x 4
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
            nn.Dropout(0.7),
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
        bbox_data = bbox_data.with_columns(pl.col("Finding Label").str.replace("Infiltrate", "Infiltration"))

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
        """Forward pass of the model
        
        TODO: test if changing the order to the front instead of attention layer on the backend makes a noticeable difference.
        """
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
        """
    
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=5),
            # transforms.Grayscale(num_output_channels=1)
        ])
        
        # Load bounding box data and data entry data
        bbox_data = pl.read_csv(bbox_csv_path, new_columns=["Image Index", "Finding Label", "x", "y", "w", "h"]) 
        bbox_data = bbox_data.with_columns(pl.col("Finding Label").str.replace("Infiltrate", "Infiltration")) #The definition between the findings and bbox should match


        data_entry_data = pl.read_csv(data_entry_path)
        data_entry_data = data_entry_data.filter(pl.col("Image Index").is_in(os.listdir(root_dir)))

        #Only those actually present should be included.
        bbox_data = bbox_data.filter(pl.col("Image Index").is_in(os.listdir(root_dir)))

        # Get all unique image filenames
        # self.image_filenames = bbox_data.select("Image Index").unique()["Image Index"].to_list() 
        self.image_filenames = os.listdir(root_dir)

        # Get all unique finding labels (classes)
        self.classes =  data_entry_data.get_column('Finding Labels').str.split('|').map_elements(lambda x: [f.strip() for f in x], return_dtype=pl.List(pl.Utf8)).explode().unique()
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
            image = Image.open(img_path).convert('L') # Convert to grayscale
        
            # transformation to Torch tensor
            image = self.transform(image)  # Shape: (H, W) → (1, H, W)

            if image.ndimension() == 2:  # Should be (1, H, W) already. This should be 3 if not using batch sizes
                image = image.unsqueeze(0)  # Add batch dimension → (1, H, W)
        else:
            image = None

        # Create multi-hot encoding for labels
        # This needs  to be done with data entry
        label_tensor = torch.zeros(len(self.classes))
        if img_file in self.image_findings:
            for finding, _ in self.image_findings[img_file].items(): #BBoxes not used 
                label_tensor[self.class_to_idx[finding]] = 1
        # label_tensor = label_tensor.unsqueeze(0) # Not needed when using batches. 

        return image, label_tensor, img_file
    
    def get_labels(self) -> list:
        """Get the labels we are using"""
        return [key for key in self.class_to_idx]
    

def setup_model_and_training(dataset_xrays=None, batch_size=16, learning_rate=0.0001, model_type="simpleCNN", model_mode="single", path_multi=os.path.join("..", "data", "images_train")):
    """Sets up a model, loads data into a dataloader and defines the modeltype and the model mode according to wishes
    
    If using the multi attention model, this should be """
    # Create data loader
    if model_type == "MultiAttention":
        dataset = XrayMultiLabelDataset(path_multi)
    else:
        dataset = dataset_xrays

    # Wrap dataset in DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Get number of classes
    num_classes = len(data_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {data_loader.dataset.classes}")
    
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
        # Add class weighting in loss function to prevent overfitting on no findings
        targets = torch.zeros(len(dataset), num_classes)
        for i in range(len(dataset)):
            _, label_tensor, _ = dataset[i]
            targets[i] = label_tensor
            
        pos_counts = targets.sum(0)
        neg_counts = len(targets) - pos_counts
        pos_weight = neg_counts / pos_counts.clamp(min=1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    return model, data_loader, criterion, optimizer, device


def setup_dataloader(dataset_xrays=None, batch_size=16, num_workers=4, shuffle=True, model_type="single", path_multi=os.path.join("..", "data", "images_train")):
    """Create a dataloader"""
    if model_type == "MultiAttention":
        dataset = XrayMultiLabelDataset(path_multi)
    else:
        dataset = dataset_xrays

    # Wrap dataset in DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
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


def train_multilabel_model(model, data_loader, criterion, optimizer, device, num_epochs=10, data_loader_test=None, basename="CNNMultiAttent"):
    """Should be combined with original function"""
    run_name = f"{basename}{time.strftime('%Y%m%d%H%M')}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    model.train()
    
    # Build class mapping (finding name to index)
    class_mapping = {class_name: idx for idx, class_name in enumerate(data_loader.dataset.classes)}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for idx, (inputs, labels, filenames) in enumerate(data_loader):
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

            # You can also log batch-level metrics if needed
            writer.add_scalar('batch_loss', loss.detach().item(), epoch * len(data_loader) + idx)
        
        epoch_loss = running_loss / len(data_loader)
        writer.add_scalar('training_loss', epoch_loss, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f'param/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'grad/{name}', param.grad, epoch)


        print(f"Data loader length: {len(data_loader)}")
        # epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # If we have validation data
        if data_loader_test:
            val_loss, val_acc, val_f1 = validate(model, data_loader_test, criterion, device, class_mapping)
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)
            writer.add_scalar('Validation/F1_Score', val_f1, epoch)

    
    model_description = get_model_description(model, None, device=device)
    writer.add_text('Model/Architecture', model_description, 0)

    writer.close()
    return model


"""General functions begin here"""
def get_model_description(model, input_shape=None, device=None):
    """
    Generate a comprehensive description of a PyTorch model including architecture,
    layer details, parameters, and hyperparameters.
    
    Args:
        model: The PyTorch model
        input_shape: Optional tuple of input dimensions (excluding batch dimension)
                    Example: (3, 224, 224) for RGB images
    
    Returns:
        String containing formatted model description in Markdown
    """
    # Get a string representation of the model architecture
    model_str = str(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get detailed information about each layer
    layer_info = []
    
    for name, module in model.named_modules():
        # Skip the model itself and container modules with no parameters
        if module is model or (not list(module.parameters()) and 
                             not isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, 
                                                  nn.Flatten, nn.Softmax, nn.Sigmoid))):
            continue
            
        # Get module type
        module_type = module.__class__.__name__
        
        # Get module-specific hyperparameters
        params = {}
        
        # Convolutional layers
        if isinstance(module, nn.Conv1d):
            params = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "bias": module.bias is not None
            }
        elif isinstance(module, nn.Conv2d):
            params = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "bias": module.bias is not None
            }
        # Pooling layers
        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
            params = {
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding
            }
        # Batch normalization
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            params = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine,
                "track_running_stats": module.track_running_stats
            }
        # Linear layers
        elif isinstance(module, nn.Linear):
            params = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None
            }
        # Dropout layers
        elif isinstance(module, nn.Dropout):
            params = {"p": module.p}
        elif isinstance(module, nn.Dropout2d):
            params = {"p": module.p}
        # Activation functions
        elif isinstance(module, nn.ReLU):
            params = {"inplace": module.inplace}
        elif isinstance(module, nn.LeakyReLU):
            params = {"negative_slope": module.negative_slope, "inplace": module.inplace}
        # Recurrent layers
        elif isinstance(module, nn.LSTM):
            params = {
                "input_size": module.input_size,
                "hidden_size": module.hidden_size,
                "num_layers": module.num_layers,
                "bias": module.bias,
                "batch_first": module.batch_first,
                "dropout": module.dropout,
                "bidirectional": module.bidirectional
            }
        elif isinstance(module, nn.GRU):
            params = {
                "input_size": module.input_size,
                "hidden_size": module.hidden_size,
                "num_layers": module.num_layers,
                "bias": module.bias,
                "batch_first": module.batch_first,
                "dropout": module.dropout,
                "bidirectional": module.bidirectional
            }
        elif isinstance(module, nn.Embedding):
            params = {
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx
            }
            
        # Calculate parameters for this layer
        layer_params = sum(p.numel() for p in module.parameters())
        trainable_layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        layer_info.append({
            "name": name,
            "type": module_type,
            "params": params,
            "param_count": layer_params,
            "trainable_param_count": trainable_layer_params
        })
    
    # Format everything nicely using Markdown for TensorBoard
    output = StringIO()
    output.write("# Model Architecture Summary\n\n")
    
    output.write("## Model Structure\n```\n")
    output.write(model_str)
    output.write("\n```\n\n")
    
    output.write("## Parameter Counts\n")
    output.write(f"Total parameters: {total_params:,}\n")
    output.write(f"Trainable parameters: {trainable_params:,}\n")
    output.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n\n")
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 parameters
    output.write(f"Approximate model size (in memory): {model_size_mb:.2f} MB\n\n")
    
    # Layer details in a table format
    output.write("## Layer Details\n\n")
    output.write("| Layer | Type | Parameters | Trainable | Details |\n")
    output.write("|-------|------|------------|-----------|--------|\n")
    
    for layer in layer_info:
        # Format parameters as pretty JSON
        param_str = json.dumps(layer["params"], indent=2).replace('\n', '<br>').replace(' ', '&nbsp;')
        output.write(f"| {layer['name']} | {layer['type']} | {layer['param_count']:,} | {layer['trainable_param_count']:,} | {param_str} |\n")
    
    # If input shape is provided, calculate output shapes
    if input_shape is not None:
        try:
            output.write("\n## Layer Output Shapes\n\n")
            output.write("| Layer | Output Shape |\n")
            output.write("|-------|-------------|\n")
            
            # Register a forward hook to capture output sizes
            output_shapes = {}
            
            def hook_fn(module, input, output):
                # Convert output tensor or tuple of tensors to a string representation of their shapes
                if isinstance(output, torch.Tensor):
                    output_shapes[module] = tuple(output.shape)
                elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                    output_shapes[module] = tuple(tuple(o.shape) for o in output)
            
            hooks = []
            for name, module in model.named_modules():
                if module is not model:  # Skip the model itself
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass with dummy input
            dummy_input = torch.zeros(1, *input_shape)
            model(dummy_input, device)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Output the shapes
            for name, module in model.named_modules():
                if module is not model and module in output_shapes:
                    output.write(f"| {name} | {output_shapes[module]} |\n")
                    
        except Exception as e:
            output.write(f"\nCould not calculate output shapes: {str(e)}\n")
    
    return output.getvalue()

def validate(model, val_loader, criterion, device, class_mapping, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        for inputs, labels, filenames in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs, filenames, class_mapping)  # Shape: [batch_size, 15]
            loss = criterion(outputs, labels)
            running_loss += loss.mean().item() * inputs.size(0)
            
            # Convert probabilities to binary predictions
            probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            preds = (probs > threshold).int()  # Thresholding at 0.5
            
            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute final loss, accuracy, and F1-score
    val_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)  # Multi-label accuracy
    f1 = f1_score(all_labels, all_preds, average='samples')  # Adjusted for multi-label

    return val_loss, accuracy, f1


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



if __name__ == "__main__":
    os.chdir("src")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Grayscale(num_output_channels=1), 
    ])
    model, data_loader, criterion, optimizer, device = setup_model_and_training(None, model_type="MultiAttention", batch_size=16, model_mode="multi", learning_rate=0.00001)
    # data_loader_test = md.XrayMultiLabelDataset(os.path.join("data", "images_test"))
    # model = md.train_model(model, data_loader, criterion, optimizer, device, num_epochs=5)
    data_loader_test = setup_dataloader(None, path_multi=os.path.join("..", "data", "images_test"), model_type="MultiAttention") # dataset_xrays_test
    model = train_multilabel_model(model, data_loader, criterion, optimizer, device, num_epochs=2)