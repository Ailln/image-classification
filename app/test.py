import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

from models import shuffle_net_v2
from utils.conf_utils import get_conf
from utils.data_utils import read_json

parser = argparse.ArgumentParser(description="Image Classification Test")
parser.add_argument("--conf", default="./conf/dev.yaml", type=str, help="conf path")
parser.add_argument("--model_path", default="./save/epoch_11-train_acc_94.885-validate_acc_91.86.ckpt", type=str,
                    help="best model path")
args = parser.parse_args()

conf = get_conf(args.conf)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
print(">> Building model..")
net = shuffle_net_v2.ShuffleNetV2(2)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print(">> Resuming from checkpoint...")
checkpoint = torch.load(args.model_path, map_location="cpu")
net.load_state_dict(checkpoint)


def test(img_path):
    net.eval()
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
        outputs = net(inputs)
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
    for i in range(1, 10):
        image_path = f"{test_path}/{i}.jpg"
        print(i, test(image_path))
