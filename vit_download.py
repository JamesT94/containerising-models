from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

# Load a pretrained ViT model and its feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Save the model and feature extractor
model.save_pretrained('vit_model')
feature_extractor.save_pretrained('vit_feature_extractor')
