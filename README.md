# dogorcat

## 概要

dogorcatは、画像から犬か猫かを判定するWebアプリケーションです。DjangoフレームワークとPyTorchによる機械学習モデルを利用しています。

## 主な機能
- 画像アップロードによる犬・猫の判定
- 判定結果の表示

## ディレクトリ構成
- `dogorcatvenv/` : Python仮想環境
- `dogorcat/` : Djangoプロジェクト本体
  - `ml_models/` : 機械学習モデル関連
  - `prediction/` : 判定アプリケーション

## セットアップ手順
1. 仮想環境の有効化
   ```
   .\dogorcatvenv\Scripts\activate
   ```
2. 依存パッケージのインストール
   ```
   pip install -r dogorcatvenv/requirements.txt
   ```
3. マイグレーションの実行
   ```
   python dogorcatvenv/dogorcat/manage.py migrate
   ```
4. サーバーの起動
   ```
   python dogorcatvenv/dogorcat/manage.py runserver
   ```

## モデルファイル
- `ml_models/cnn_dogcat_weights.pth` : PyTorchで学習済みの重みファイル

## テスト
pytestを利用してテストを実行できます。
```
pytest
```

## ライセンス
このプロジェクトはMITライセンスです。
