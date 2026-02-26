import torch
from .model import CNN
from django.conf import settings
import os

model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'cnn_dogcat_weights.pth')

#モデルをロード
model = CNN(2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

