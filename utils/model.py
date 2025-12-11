import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from utils.config import SR,MEL_BINS

sys.path.append("audioset_tagging_cnn/pytorch")
from models import Cnn14 as Cnn14Base


class MarineClassifier:
    def __init__(self, num_classes, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = None
        
    def load_pretrained(self):
        """Load pretrained CNN14 and adapt for our classes"""
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
            map_location="cpu"
        )
        
        self.model = Cnn14Base(sample_rate=SR, window_size=1024, hop_size=320,
                               mel_bins=MEL_BINS, fmin=50, fmax=14000, classes_num=527)
        self.model.load_state_dict(checkpoint['model'])
        
        # Freeze and modify
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc1.parameters():
            p.requires_grad = True
        
        self.model.fc_audioset = nn.Linear(self.model.fc1.out_features, self.num_classes)
        for p in self.model.fc_audioset.parameters():
            p.requires_grad = True
            
        self.model = self.model.to(self.device)
        return self.model
    
    def load_finetuned(self, path):
        """Load already trained model"""
        self.model = Cnn14Base(sample_rate=SR, window_size=1024, hop_size=320,
                               mel_bins=MEL_BINS, fmin=50, fmax=14000, classes_num=527)
        self.model.fc_audioset = nn.Linear(self.model.fc1.out_features, self.num_classes)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)
        return self.model
    
    def train_epoch(self, loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            out = self.model(x)
            preds = out["clipwise_output"]

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(preds, dim=1)
            total_correct += (predicted == y).sum().item()
            total_samples += x.size(0)

        return total_loss / total_samples, total_correct / total_samples
    
    def evaluate(self, loader, criterion):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)
                preds = out["clipwise_output"]

                loss = criterion(preds, y)

                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(preds, dim=1)
                total_correct += (predicted == y).sum().item()
                total_samples += x.size(0)

        return total_loss / total_samples, total_correct / total_samples
    
    def save(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)