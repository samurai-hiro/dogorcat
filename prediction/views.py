from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from .forms import ImageUpLoadForm
from PIL import Image
import torch
import torchvision.transforms as transforms
from ml_models.load_model import model

#画像前処理インスタンス
transform: transforms.Compose = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
class_names: list[str] = ['猫', '犬']

def predict(request: HttpRequest) -> HttpResponse:
    if request.method == 'GET':
        form = ImageUpLoadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUpLoadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            img = Image.open(image).convert('RGB')
            x = transform(img)
            x = x.unsqueeze(0)

            #予測
            with torch.no_grad():
                outputs = model(x)
                probalities = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs,dim=1)
                pred_class = class_names[pred.item()]
                confidence = float(probalities[0][pred.item()] * 100)
            
            #画像を取得して表示
            img_data = request.POST.get('img_data')
            return render(request, 'home.html',{'form': form,
                                                'prediction': pred_class,
                                                'confidence': confidence,
                                                'img_data': img_data})
        else:
            # 無効な画像の場合のエラーハンドリング
            form = ImageUpLoadForm()
            return render(request, 'home.html',
                           {'form':form, 'error':'無効な画像です。'})
        
def disclaimer(request):
    return render(request, 'disclaimer.html')