"""
This file combines the code in one file.
dataset, model, train, test, inference, etc.

Yongming, Chenxi
2025.07.12
"""

# Standard library imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np

# Third-party imports for data processing and augmentation
import albumentations as A  # Image augmentation library
import pandas as pd
import cv2  # OpenCV for image processing

# Hugging Face transformers for text processing
from transformers import AutoTokenizer, AutoModel, AutoConfig, DistilBertModel, DistilBertConfig
from tqdm import tqdm  # Progress bar
import itertools

# PyTorch Image Models for vision models
import timm

## Configuration Class
class CFG:
    """
    Configuration class containing all hyperparameters and settings for the CLIP model training.
    This centralizes all configurable parameters for easy modification.
    """
    # Debug mode - reduces dataset size for quick testing
    debug = False
    
    # Data paths
    image_path = "/home/yq/ssd/clip/flickr8k/Images"  # Path to image directory
    captions_path = "/home/yq/ssd/clip/flickr8k/"     # Path to captions file
    
    # Training hyperparameters
    batch_size = 32          # Number of samples per batch
    num_workers = 4          # Number of data loading workers
    head_lr = 1e-3          # Learning rate for projection heads
    image_encoder_lr = 1e-4  # Learning rate for image encoder
    text_encoder_lr = 1e-5   # Learning rate for text encoder
    weight_decay = 1e-3      # Weight decay for regularization
    patience = 1             # Patience for learning rate scheduler
    factor = 0.8             # Factor for reducing learning rate
    epochs = 4               # Number of training epochs
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model architecture parameters
    model_name = 'resnet50'                    # Image encoder model
    image_embedding = 2048                     # Image feature dimension
    text_encoder_model = "distilbert-base-uncased"  # Text encoder model
    text_embedding = 768                       # Text feature dimension
    text_tokenizer = "distilbert-base-uncased"      # Text tokenizer
    max_length = 200                           # Maximum sequence length for text

    # Model training settings
    pretrained = True  # Whether to use pretrained models for both encoders
    trainable = True   # Whether to make encoder parameters trainable
    temperature = 1.0  # Temperature parameter for contrastive learning

    # Image processing parameters
    size = 224  # Input image size (height=width)

    # Projection head parameters (used for both image and text encoders)
    num_projection_layers = 1  # Number of projection layers
    projection_dim = 256       # Dimension of projected embeddings
    dropout = 0.1              # Dropout rate

## Utility Classes and Functions
class AvgMeter:
    """
    Utility class for tracking and computing running averages of metrics.
    Useful for monitoring loss and other training metrics.
    """
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """
        Update the running average with a new value.
        
        Args:
            val: New value to add
            count: Number of samples this value represents (default: 1)
        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """String representation showing the current average."""
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """
    Extract the current learning rate from an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate (float)
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

########################################################
## Dataset Implementation
########################################################

class CLIPDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for CLIP training.
    Handles image-caption pairs and applies tokenization to text.
    
    Note: image_filenames and captions must have the same length.
    If there are multiple captions per image, image_filenames should have
    repetitive file names to match the caption count.
    """
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        Initialize the dataset.
        
        Args:
            image_filenames: List of image file names
            captions: List of corresponding captions
            tokenizer: Text tokenizer for processing captions
            transforms: Image transformation pipeline
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        
        # Tokenize all captions at initialization for efficiency
        self.encoded_captions = tokenizer(
            list(captions), 
            padding=True,           # Pad sequences to same length
            truncation=True,        # Truncate if too long
            max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - input_ids: Tokenized text input
            - attention_mask: Attention mask for text
            - image: Processed image tensor
            - caption: Original caption text
        """
        # Convert tokenized text to tensors
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # Load and process image
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = self.transforms(image=image)['image']   # Apply transformations
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # Convert to tensor and change format
        item['caption'] = self.captions[idx]  # Store original caption

        return item

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.captions)

def get_transforms(mode="train"):
    """
    Create image transformation pipeline.
    
    Args:
        mode: Either "train" or "valid" (currently same for both)
        
    Returns:
        Albumentations transformation pipeline
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),  # Resize to fixed size
                A.Normalize(max_pixel_value=255.0, always_apply=True),  # Normalize pixel values
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

########################################################
## Model Architecture
########################################################

class ImageEncoder(nn.Module):
    """
    Image encoder using a pre-trained vision model (ResNet50 by default).
    Encodes images to a fixed-size feature vector.
    """
    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        """
        Initialize the image encoder.
        
        Args:
            model_name: Name of the vision model to use
            pretrained: Whether to use pre-trained weights
            trainable: Whether to make the model parameters trainable
        """
        super().__init__()
        # Create model with no classification head (num_classes=0)
        # and global average pooling
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        
        # Set parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Image features [batch_size, feature_dim]
        """
        return self.model(x)
    
class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT.
    Encodes text sequences to fixed-size feature vectors.
    """
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Name of the transformer model to use
            pretrained: Whether to use pre-trained weights
            trainable: Whether to make the model parameters trainable
        """
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        # Set parameter trainability
        for p in self.model.parameters():
            p.requires_grad = trainable

        # Use CLS token ([0] position) as the sentence embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Tokenized input text [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text features [batch_size, feature_dim] from CLS token
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]  # Extract CLS token

class ProjectionHead(nn.Module):
    """
    Projection head that maps encoder features to a common embedding space.
    Uses a residual connection and layer normalization for better training.
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        """
        Initialize the projection head.
        
        Args:
            embedding_dim: Input feature dimension
            projection_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)  # Initial projection
        self.gelu = nn.GELU()  # GELU activation
        self.fc = nn.Linear(projection_dim, projection_dim)  # Additional linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        self.layer_norm = nn.LayerNorm(projection_dim)  # Layer normalization
    
    def forward(self, x):
        """
        Forward pass through the projection head.
        
        Args:
            x: Input features [batch_size, embedding_dim]
            
        Returns:
            Projected features [batch_size, projection_dim]
        """
        projected = self.projection(x)  # Initial projection
        x = self.gelu(projected)        # Apply GELU activation
        x = self.fc(x)                  # Additional transformation
        x = self.dropout(x)             # Apply dropout
        x = x + projected               # Residual connection
        x = self.layer_norm(x)          # Layer normalization
        return x

class CLIPModel(nn.Module):
    """
    Complete CLIP model that combines image and text encoders with projection heads.
    Implements contrastive learning loss for training.
    """
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        """
        Initialize the CLIP model.
        
        Args:
            temperature: Temperature parameter for contrastive learning
            image_embedding: Dimension of image encoder output
            text_embedding: Dimension of text encoder output
        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """
        Forward pass through the CLIP model.
        
        Args:
            batch: Dictionary containing:
                - image: Image tensors [batch_size, channels, height, width]
                - input_ids: Tokenized text [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                
        Returns:
            Contrastive learning loss (scalar tensor)
        """
        # Encode images and text to get features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        
        # Project features to common embedding space
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculate contrastive learning loss
        # Compute similarity matrix between text and image embeddings
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        
        # Compute similarity matrices for images and texts with themselves
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        
        # Create targets using softmax of average similarities
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        
        # Calculate bidirectional contrastive loss
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # Average of both directions
        
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    """
    Custom cross-entropy loss function for contrastive learning.
    
    Args:
        preds: Predicted logits [batch_size, batch_size]
        targets: Target probabilities [batch_size, batch_size]
        reduction: How to reduce the loss ('none', 'mean')
        
    Returns:
        Cross-entropy loss
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)  # Sum over classes dimension
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

########################################################
## Data Loading and Training Functions
########################################################

def make_train_valid_dfs():
    """
    Create training and validation dataframes by splitting the dataset.
    
    Returns:
        train_df: Training dataframe
        valid_df: Validation dataframe
    """
    # Load captions data
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.txt")
    
    # Get unique image filenames and create a mapping to numeric IDs
    unique_images = dataframe["image"].unique()
    image_to_id = {img: idx for idx, img in enumerate(unique_images)}
    
    # Add a numeric ID column to the dataframe
    dataframe["image_id"] = dataframe["image"].map(image_to_id)
    
    # Determine maximum image ID (for debug mode, limit to 100)
    max_id = len(unique_images) if not CFG.debug else min(100, len(unique_images))
    image_ids = np.arange(max_id)
    
    # Split data into train/validation sets (80/20 split)
    np.random.seed(42)  # For reproducibility
    valid_ids = np.random.choice(image_ids, size=int(0.2 * max_id), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    
    # Create dataframes
    train_df = dataframe[dataframe["image_id"].isin(train_ids)].reset_index(drop=True)
    valid_df = dataframe[dataframe["image_id"].isin(valid_ids)].reset_index(drop=True)
    
    return train_df, valid_df

def build_loaders(dataframe, tokenizer, mode):
    """
    Create data loaders for training or validation.
    
    Args:
        dataframe: DataFrame containing image-caption pairs
        tokenizer: Text tokenizer
        mode: Either "train" or "valid"
        
    Returns:
        DataLoader for the specified mode
    """
    transforms = get_transforms(mode)
    dataset = CLIPDataset(
        image_filenames=dataframe["image"].values,
        captions=dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,  # Only shuffle training data
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """
    Train the model for one epoch.
    
    Args:
        model: CLIP model
        train_loader: Training data loader
        optimizer: Optimizer for updating parameters
        lr_scheduler: Learning rate scheduler
        step: When to step the scheduler ("batch" or "epoch")
        
    Returns:
        Average training loss for the epoch
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
        # Move batch to device (except captions which are strings)
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        
        # Forward pass
        loss = model(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step scheduler if using batch-level scheduling
        if step == "batch":
            lr_scheduler.step()

        # Update loss meter
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        # Update progress bar
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    return loss_meter

def valid_epoch(model, valid_loader):
    """
    Validate the model for one epoch.
    
    Args:
        model: CLIP model
        valid_loader: Validation data loader
        
    Returns:
        Average validation loss for the epoch
    """
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    for batch in tqdm_object:
        # Move batch to device
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        
        # Forward pass (no gradients needed for validation)
        with torch.no_grad():
            loss = model(batch)

        # Update loss meter
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        # Update progress bar
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    
    return loss_meter

########################################################
## Main Training Function
########################################################

def main():
    """
    Main training function that orchestrates the entire training process.
    """
    # Prepare data
    print("Preparing data...")
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, "train")
    valid_loader = build_loaders(valid_df, tokenizer, "valid")
    
    # Initialize model
    print("Initializing model...")
    model = CLIPModel().to(CFG.device)
    
    # Set up optimizer with different learning rates for different components
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(),
            model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = optim.AdamW(params, weight_decay=0.)
    
    # Learning rate scheduler
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=CFG.patience, factor=CFG.factor
    )
    
    step = "epoch"  # Step scheduler after each epoch
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch} / {CFG.epochs}")
        
        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_schedule, step)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            
        # Save best model
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")
            
        # Step learning rate scheduler
        lr_schedule.step(valid_loss.avg)
        
        # Print epoch results
        print(f"Train Loss: {train_loss.avg:.4f}, Valid Loss: {valid_loss.avg:.4f}")
        
if __name__ == "__main__":
    main()