import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.models import vgg11_bn
from torchvision.models import resnet18
from torchvision.models import mobilenet
from torchvision.models import shufflenetv2
from PIL import Image

from models import crnn
from utils.conf_utils import get_conf
from utils.data_utils import read_json

parser = argparse.ArgumentParser(description="Image Classification Test")
parser.add_argument("--conf", default="./conf/dev.yaml", type=str, help="conf path")
parser.add_argument("--model_path", default="./save/mobilenet-epoch_2-train_acc_97.625-validate_acc_97.28.ckpt",
                    type=str, help="best model path")
args = parser.parse_args()

conf = get_conf(args.conf)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = conf["parameters"]["model_name"]
print(f">> Building {model_name} model...")
model_dict = {
    "shufflenet": shufflenetv2.shufflenet_v2_x1_0(pretrained=True),
    "mobilenet": mobilenet.mobilenet_v2(pretrained=True),
    "vgg": vgg11_bn(pretrained=True),
    "resnet": resnet18(pretrained=True),
    "crnn": crnn.CRNN()
}
model = model_dict[model_name]
model = model.to(device)

if device == "cuda":
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Load checkpoint.
print(">> Resuming from checkpoint...")
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))


def test(img_path):
    model.eval()
    with torch.no_grad():
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        img = Image.open(img_path)
        inputs = transform_train(img).unsqueeze(0)
        outputs = model(inputs)
        idx = torch.argmax(outputs).cpu().numpy()
        class_to_idx = read_json(conf["dataset"]["class_to_idx"])
        result = None
        for key, value in class_to_idx.items():
            if idx == value:
                result = key
                break

        return result


if __name__ == "__main__":
    test_path = conf["dataset"]["test"]
    for i in range(10):
        image_path = f"{test_path}/{i+1}.jpg"
        print(image_path, test(image_path))
