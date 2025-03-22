import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
import pandas as pd



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

def setup_model_and_training(dataset_xrays, batch_size=16, learning_rate=0.001, model_type="simpleCNN"):
    # Create data loader
    data_loader = DataLoader(dataset_xrays, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Get number of classes
    num_classes = len(dataset_xrays.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {dataset_xrays.classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "simpleCNN":
        model = SimpleXrayCNN(num_classes=num_classes)
    elif model_type == "ResNet18":
        model = FineTunedResNet18(num_classes=num_classes)
    elif model_type == "ResNet50":
        model = FineTunedResNet50(num_classes=num_classes)
    elif model_type == "DenseNet121":
        model = FineTunedDenseNet121(num_classes=num_classes)
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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

def predict_single(model, image_tensor, device, class_names=None, return_probs=False):
    """
    Make a prediction for a single image.
    
    Args:
        model (nn.Module): The trained model
        image_tensor (torch.Tensor): Image tensor (C x H x W)
        device (torch.device): Device to run prediction on
        class_names (list): Optional list of class names
        return_probs (bool): Whether to return probabilities
        
    Returns:
        dict: Prediction results
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Get prediction
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        result = {
            'predicted_idx': predicted_idx.item(),
            'confidence': confidence.item(),
        }
        
        if class_names:
            result['predicted_class'] = class_names[predicted_idx.item()]
        
        if return_probs:
            result['probabilities'] = probabilities.squeeze().cpu().numpy()
            if class_names:
                result['class_probabilities'] = {
                    class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities.squeeze())
                }
        
        return result

def predict_batch(model, data_loader, device, max_samples=None):
    """
    Make predictions for a batch of images.
    
    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to run prediction on
        max_samples (int): Maximum number of samples to predict
        
    Returns:
        tuple: (all_predictions, all_labels, all_confidences)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(data_loader, desc="Predicting")):
            # Check if we've reached the maximum number of samples
            if max_samples and i * data_loader.batch_size >= max_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_confidences), np.vstack(all_probs)

def evaluate_model(model, data_loader, device, class_names=None, max_samples=None, visualize=True):
    """
    Comprehensive model evaluation with detailed metrics.
    
    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to run evaluation on
        class_names (list): Optional list of class names
        max_samples (int): Maximum number of samples to evaluate
        visualize (bool): Whether to visualize results
        
    Returns:
        dict: Evaluation metrics
    """
    start_time = time.time()
    
    # Get predictions
    y_pred, y_true, confidences, probabilities = predict_batch(
        model, data_loader, device, max_samples)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'samples_evaluated': len(y_true),
        'avg_confidence': np.mean(confidences),
        'evaluation_time': time.time() - start_time
    }
    
    # Calculate per-class metrics if we have class names
    if class_names:
        # Use weighted average for multi-class
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_indices = (y_true == i)
            if np.any(class_indices):
                class_pred = y_pred[class_indices]
                class_accuracy = np.mean(class_pred == i)
                class_samples = np.sum(class_indices)
                class_metrics[class_name] = {
                    'accuracy': class_accuracy,
                    'samples': class_samples,
                    'confidence': np.mean(confidences[y_pred == i]) if np.any(y_pred == i) else 0
                }
        
        metrics['class_metrics'] = class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Visualize if requested
        if visualize:
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
            
            # Plot class accuracies
            class_accs = [metrics['class_metrics'][c]['accuracy'] for c in class_names]
            plt.figure(figsize=(12, 6))
            bars = plt.bar(class_names, class_accs)
            plt.axhline(y=accuracy, color='r', linestyle='-', label=f'Overall Accuracy: {accuracy:.3f}')
            plt.ylabel('Accuracy')
            plt.title('Per-Class Accuracy')
            plt.xticks(rotation=45, ha='right')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    # Print metrics summary
    print("======= Evaluation Results =======")
    print(f"Accuracy: {accuracy:.4f}")
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    print(f"Samples Evaluated: {metrics['samples_evaluated']}")
    print(f"Evaluation Time: {metrics['evaluation_time']:.2f} seconds")
    
    return metrics

def visualize_predictions(model, dataset, device, indices=None, num_examples=5, class_names=None):
    """
    Visualize model predictions on example images.
    
    Args:
        model (nn.Module): The trained model
        dataset: The dataset containing images
        device (torch.device): Device to run prediction on
        indices (list): Optional specific indices to visualize
        num_examples (int): Number of examples to visualize
        class_names (list): Optional list of class names
    """
    model.eval()
    
    # If no specific indices provided, randomly select some
    if indices is None:
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    else:
        indices = indices[:num_examples]  # Limit to num_examples
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 4))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get the image and label
        image, label = dataset[idx]
        
        # Make prediction
        result = predict_single(model, image, device, class_names)
        
        # Convert image for display
        if isinstance(image, torch.Tensor):
            # Denormalize if needed
            if image.min() < 0:
                image = image * 0.5 + 0.5
            
            # Convert to numpy for display
            img_display = image.cpu().numpy().transpose((1, 2, 0))
            
            # If grayscale, squeeze
            if img_display.shape[2] == 1:
                img_display = img_display.squeeze()
        else:
            img_display = image
        
        # Display image
        axes[i].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        
        # Set title
        pred_idx = result['predicted_idx']
        conf = result['confidence']
        
        if class_names:
            true_class = class_names[label]
            pred_class = class_names[pred_idx]
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.2f}"
        else:
            title = f"True: {label}\nPred: {pred_idx}\nConf: {conf:.2f}"
        
        # Color title based on correctness
        color = 'green' if pred_idx == label else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_misclassifications(model, data_loader, device, class_names=None, max_samples=100):
    """
    Analyze and summarize misclassifications.
    
    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to run evaluation on
        class_names (list): Optional list of class names
        max_samples (int): Maximum number of samples to analyze
        
    Returns:
        DataFrame: Summary of misclassifications
    """
    model.eval()
    
    misclassifications = []
    samples_processed = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            batch_size = inputs.size(0)
            samples_processed += batch_size
            
            # Check if we've reached the maximum
            if samples_processed > max_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, 1)
            
            # Find misclassifications
            misclassified_indices = (predictions != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_indices:
                true_label = labels[idx].item()
                pred_label = predictions[idx].item()
                conf = confidence[idx].item()
                
                # Get top-3 predictions
                top3_values, top3_indices = torch.topk(probabilities[idx], min(3, outputs.size(1)))
                top3 = [(top3_indices[j].item(), top3_values[j].item()) for j in range(len(top3_indices))]
                
                # Store misclassification info
                misclass_info = {
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': conf,
                    'top3_predictions': top3
                }
                
                if class_names:
                    misclass_info['true_class'] = class_names[true_label]
                    misclass_info['predicted_class'] = class_names[pred_label]
                    misclass_info['top3_classes'] = [(class_names[idx], val) for idx, val in top3]
                
                misclassifications.append(misclass_info)
    
    # Convert to DataFrame
    if not misclassifications:
        print("No misclassifications found in the analyzed samples!")
        return None
    
    df = pd.DataFrame(misclassifications)
    
    # Print summary
    print(f"Found {len(misclassifications)} misclassifications in {samples_processed} samples")
    print(f"Misclassification rate: {len(misclassifications)/samples_processed:.4f}")
    
    if class_names:
        # Group by true class and predicted class
        confusion_pairs = df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
        confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
        
        print("\nTop misclassification pairs:")
        print(confusion_pairs.head(10))
        
        # Group by true class
        true_class_counts = df.groupby('true_class').size().reset_index(name='count')
        true_class_counts = true_class_counts.sort_values('count', ascending=False)
        
        print("\nMost frequently misclassified classes:")
        print(true_class_counts.head(5))
    
    # Plot confidence distribution for misclassifications
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, alpha=0.7)
    plt.axvline(x=df['confidence'].mean(), color='r', linestyle='--', 
                label=f'Mean Confidence: {df["confidence"].mean():.3f}')
    plt.title('Confidence Distribution for Misclassifications')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    return df

def find_hard_examples(model, data_loader, device, n=5, class_names=None):
    """
    Find the hardest examples to classify based on confidence.
    
    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to run prediction on
        n (int): Number of examples to find
        class_names (list): Optional list of class names
        
    Returns:
        tuple: Indices of hard examples and their metadata
    """
    model.eval()
    
    all_confidences = []
    all_correct = []
    all_indices = []
    batch_offset = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Finding hard examples"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, 1)
            
            # Check correctness
            correct = (predictions == labels)
            
            # Store results with indices
            all_confidences.append(confidence.cpu().numpy())
            all_correct.append(correct.cpu().numpy())
            
            # Store indices adjusted for batch position
            batch_indices = np.arange(batch_offset, batch_offset + inputs.size(0))
            all_indices.append(batch_indices)
            batch_offset += inputs.size(0)
    
    # Combine results
    confidences = np.concatenate(all_confidences)
    correct = np.concatenate(all_correct)
    indices = np.concatenate(all_indices)
    
    # Define hard examples:
    # 1. Misclassified with high confidence
    # 2. Correctly classified but with low confidence
    
    # Find misclassified with highest confidence
    misclassified = ~correct
    if np.any(misclassified):
        mis_conf = confidences[misclassified]
        mis_idx = indices[misclassified]
        
        # Sort by confidence (descending)
        mis_sort_idx = np.argsort(mis_conf)[::-1]
        hard_mis = mis_idx[mis_sort_idx[:n]]
        hard_mis_conf = mis_conf[mis_sort_idx[:n]]
        hard_mis_meta = [(idx, conf, False) for idx, conf in zip(hard_mis, hard_mis_conf)]
    else:
        hard_mis_meta = []
    
    # Find correctly classified with lowest confidence
    correctly_classified = correct
    if np.any(correctly_classified):
        cor_conf = confidences[correctly_classified]
        cor_idx = indices[correctly_classified]
        
        # Sort by confidence (ascending)
        cor_sort_idx = np.argsort(cor_conf)
        hard_cor = cor_idx[cor_sort_idx[:n]]
        hard_cor_conf = cor_conf[cor_sort_idx[:n]]
        hard_cor_meta = [(idx, conf, True) for idx, conf in zip(hard_cor, hard_cor_conf)]
    else:
        hard_cor_meta = []
    
    # Combine and sort by 'hardness' (low confidence for correct, high confidence for incorrect)
    hard_examples = hard_mis_meta + hard_cor_meta
    hard_examples.sort(key=lambda x: x[1] if not x[2] else 1-x[1], reverse=True)
    
    # Limit to n examples
    hard_examples = hard_examples[:n]
    
    # Print summary
    print("===== Hard Examples Summary =====")
    for i, (idx, conf, is_correct) in enumerate(hard_examples):
        status = "Correctly classified but low confidence" if is_correct else "Misclassified with high confidence"
        print(f"{i+1}. Index {idx}: {status} (confidence: {conf:.4f})")
    
    return hard_examples