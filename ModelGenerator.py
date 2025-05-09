import numpy as np 
import pandas as pd 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

import Library as lb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

class CNN(nn.Module):
	def __init__(self, num_classes, in_channels=3):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d( in_channels=in_channels
							  , out_channels=16
							  , kernel_size=(3,3))
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d( in_channels=16
							  , out_channels=32
							  , kernel_size=(3,3))
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d(2,2)

		self.fc1 = nn.Linear(32 * 54 * 54, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = self.pool2(x)

		x = torch.flatten(x,1)
		x = self.fc1(x)
		return x

if __name__ == '__main__' :
	path = "../dataset"
	train_percent = 0.8
	batch_size = 32
	device = ""

	data_transform = transforms.Compose([ transforms.Resize(256)
										, transforms.CenterCrop(224)
										, transforms.ToTensor()
										, transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


	full_dataset = datasets.ImageFolder(path, transform = data_transform)

	train_size = int(train_percent * len(full_dataset))
	test_size = len(full_dataset) - train_size

	train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


	if (torch.cuda.is_available()):
		device = "cuda"
	else :
		device = "cpu"
	device = torch.device(device)
	print(device)


	model = CNN(num_classes=len(full_dataset.classes), in_channels=3).to(device)

	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	criterion = nn.CrossEntropyLoss()

	num_epoches = 100 #50
	patience = 20

	best_val_loss = np.inf
	counter = 0

	train_losses = []
	val_losses = []
	train_accuracies = []
	val_accuracies =[]

	train_precisions = []
	val_precisions = []
	train_recalls = []
	val_recalls = []
	train_f1s = []
	val_f1s = []

	for epoch in range(num_epoches):
		train_loss = 0.0
		correct_train = 0
		total_train = 0
		all_train_preds = []
		all_train_labels = []
		model.train()

		for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epoches} : Training"):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			train_loss += loss.item() * images.size(0)

			_, predicted = torch.max(outputs.data, 1)
			total_train += labels.size(0)
			correct_train += (predicted == labels).sum().item()

			all_train_preds.extend(predicted.cpu().numpy())
			all_train_labels.extend(labels.cpu().numpy())

		train_loss = train_loss / len(train_loader.dataset)
		train_accuracy = correct_train / total_train
		train_precision = precision_score(all_train_labels, all_train_preds, average="macro", zero_division = 1)
		train_recall = recall_score(all_train_labels, all_train_preds, average="macro")
		train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

		train_losses.append(train_loss)
		train_accuracies.append(train_accuracy)
		train_precisions.append(train_precision)
		train_recalls.append(train_recall)
		train_f1s.append(train_f1)

		val_loss = 0.0
		correct_val = 0
		total_val = 0
		all_val_preds = []
		all_val_labels = []
		model.eval()

		with torch.no_grad():
			for images, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} / {num_epoches} : Validation"):
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				val_loss += loss.item() * images.size(0)

				_, predicted = torch.max(outputs.data, 1)
				total_val += labels.size(0)
				correct_val += (predicted == labels).sum().item()

				all_val_preds.extend(predicted.cpu().numpy())
				all_val_labels.extend(labels.cpu().numpy())

		val_loss = val_loss / len(test_loader.dataset)
		val_accuracy = correct_val / total_val
		val_precision = precision_score(all_val_labels, all_val_preds, average="macro", zero_division = 1)
		val_recall = recall_score(all_val_labels, all_val_preds, average="macro")
		val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")

		val_losses.append(val_loss)
		val_accuracies.append(val_accuracy)
		val_precisions.append(val_precision)
		val_recalls.append(val_recall)
		val_f1s.append(val_f1)

		print(f"Epoch [{epoch + 1} / {num_epoches}], Train Loss : {train_loss:.4f}, Validation Loss : {val_loss:.4f}")
		print(f"Train Accuracy : {train_accuracy:.4f}, Validation Accuracy : {val_accuracy:.4f}")
		print(f"Train Precision : {train_precision:.4f}, Recall : {train_recall:.4f}, F1 : {train_f1:.4f}")
		print(f"Validation Precision : {val_precision:.4f}, Recall : {val_recall:.4f}, F1 : {val_f1:.4f}")

		if val_loss < best_val_loss :
			best_val_loss = val_loss
			counter = 0
		else:
			counter += 1
			if (counter >= patience):
				print("Premature termination is triggered")
				break
	epochs_range = range(1, len(train_losses) + 1)


	model.eval()

	correct = 0
	total = 0
	y_true = []
	x_pred = []

	with torch.no_grad():
		for images, labels in test_loader :
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			y_true.extend(labels.cpu().numpy())
			x_pred.extend(predicted.cpu().numpy())

	accuracy = 100 * correct / total
	print("Accuracy of the model on the test images : {:.2f}".format(accuracy))

	torch.save(model.state_dict(), "cnn_model.pth")