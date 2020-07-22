# image-classification

`图像分类任务` 是计算机视觉的基础任务之一，通过学习图形分类可以快速的掌握计算机视觉的基础知识。

## 1 数据集

我们使用的是一个二分类的数据集【猫】和【狗】，你可以在 `Kaggle` 上下载到这个 [数据集](https://www.kaggle.com/c/dogs-vs-cats/) 。

## 2 快速上手

```bash
# 获取代码
git clone https://github.com/Ailln/image-classification.git
cd image-classification

# 安装依赖
pip install -r requirements.txt

# 根据上文中数据集链接下载数据，并将数据放入 /datas 下的对应文件夹中
# 训练模型
python -m app.train

# 在 /save 中找到训练效果最好的模型，记录模型路径
# 开始测试
python -m app.test --model_path $your_best_model_path

# 开启图片分类 API
python -m app.api
# 发送一张图片进行测试
curl -F "upload=@/your/image/path/1.jpg; filename=1.jpg" http://127.0.0.1:8000/api/image_classification
```

## 3 网络模型

- [X] VGG
- [X] ResNet
- [X] ShuffleNet
- [x] MobileNet

## 4 许可

[MIT LICENSE](./LICENSE)