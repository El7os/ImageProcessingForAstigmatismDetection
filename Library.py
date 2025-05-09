
import numpy as np
from torchvision import models, transforms
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 


def Plot_Bar(dataset_labels, dataset_sizes, title, xlabel, ylabel):
	plt.figure(figsize=(8,6))
	plt.bar(dataset_labels, dataset_sizes, color=["red", "green"])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, size in enumerate(dataset_sizes):
		plt.text(i, size + 100, str(size), ha="center", va="center")
	plt.show()

def Plot_ClassDistribution(dataset, dataset_name, class_names, title):
	class_counts = {}
	for _, label in dataset:
		class_name = class_names[label]
		class_counts[class_name] = class_counts.get(class_name, 0) + 1

	colors = ["skyblue", "lightgreen", "orange", "mediumorchid"]
	num_classes = len(class_names)

	plt.figure(figsize=(8,6))
	plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90, colors=colors[:num_classes])
	plt.title(f'Class Distribution in {dataset_name} Set')
	plt.axis("equal")
	plt.show();

def set_seed(seed = 42):
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Enhanced data augmentation
def get_transforms():
    # Training transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms - just resize and normalize
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


# Transfer learning model using ResNet50
def get_resnet_model(num_classes):
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers to preserve learned features
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model


# Create ensemble model
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
                          
						  

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device='gpu', patience=20):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    counter = 0
    
    metrics = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': [],
        'train_precisions': [], 'val_precisions': [],
        'train_recalls': [], 'val_recalls': [],
        'train_f1s': [], 'val_f1s': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = correct / total
            
            precision = precision_score(all_labels, all_preds, average="macro", zero_division=1)
            recall = recall_score(all_labels, all_preds, average="macro")
            f1 = f1_score(all_labels, all_preds, average="macro")
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            
            # Save metrics
            if phase == 'train':
                metrics['train_losses'].append(epoch_loss)
                metrics['train_accuracies'].append(epoch_acc)
                metrics['train_precisions'].append(precision)
                metrics['train_recalls'].append(recall)
                metrics['train_f1s'].append(f1)
            else:
                metrics['val_losses'].append(epoch_loss)
                metrics['val_accuracies'].append(epoch_acc)
                metrics['val_precisions'].append(precision)
                metrics['val_recalls'].append(recall)
                metrics['val_f1s'].append(f1)
                
                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        if counter >= patience:
            break
            
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

# Evaluate the model on test data
def evaluate_model(model, test_loader, device):
    model.eval()
    
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report:")
    target_names = ["Astigmatism", "Cataract", "Diabetic Retinopathy", "Normal"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["Astigmatism", "Cataract", "Diabetic Retinopathy", "Normal"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, y_true, y_pred

def save_metrics_plot(metrics, save_path='training_metrics.png'):
    epochs_range = range(1, len(metrics['train_losses']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, metrics['train_losses'], label='Training Loss')
    plt.plot(epochs_range, metrics['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, metrics['train_accuracies'], label='Training Accuracy')
    plt.plot(epochs_range, metrics['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, metrics['train_precisions'], label='Training Precision')
    plt.plot(epochs_range, metrics['val_precisions'], label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, metrics['train_f1s'], label='Training F1')
    plt.plot(epochs_range, metrics['val_f1s'], label='Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
