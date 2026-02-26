import pytest
from django.urls import reverse
from django.test import Client

@pytest.mark.django_db
def test_home_view_get():
    client = Client()
    url = reverse('predict') # URL名が 'predict' であることを想定
    response = client.get(url)
    assert response.status_code == 200
    assert '犬、猫の画像判定をします' in response.content.decode('utf-8')


import io
from PIL import Image

@pytest.mark.django_db
def test_home_view_post_empty():
    client = Client()
    url = reverse('predict')
    response = client.post(url, {})
    assert response.status_code == 200
    assert '無効な画像です。' in response.content.decode('utf-8') or 'error' in response.content.decode('utf-8')

@pytest.mark.django_db
def test_home_view_post_image():
    client = Client()
    url = reverse('predict')
    # ダミー画像を生成
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    # フォームのフィールド名が 'image' であることを想定
    response = client.post(url, {'image': img_bytes}, format='multipart')
    assert response.status_code == 200
    # 判定結果や自信度などの文字列が含まれているか確認
    content = response.content.decode('utf-8')
    assert '判定結果' in content or 'prediction' in content