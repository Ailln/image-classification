import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import shuffle_net_v2
from utils.conf_utils import get_conf
from utils.other_utils import progress_bar
from utils.data_utils import save_json


parser = argparse.ArgumentParser(description="Image Classification Train")
parser.add_argument("--conf", default="./conf/dev.yaml", type=str, help="conf path")
args = parser.parse_args()

conf = get_conf(args.conf)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(">> Preparing data...")
transform_list = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.ImageFolder(root=conf["dataset"]["train"], transform=transform_list)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=conf["parameters"]["batch_size"], shuffle=True,
                                           num_workers=4)

print(f"## class to idx: {train_set.class_to_idx}")
save_json(train_set.class_to_idx, conf["dataset"]["class_to_idx"])

validate_set = torchvision.datasets.ImageFolder(root=conf["dataset"]["validate"], transform=transform_list)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=conf["parameters"]["batch_size"], shuffle=True,
                                              num_workers=4)

assert train_set.class_to_idx == validate_set.class_to_idx

# Model
print(">> Building model...")
net = shuffle_net_v2.ShuffleNetV2(2)
net = net.to(device)

if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=conf["parameters"]["learning_rate"], momentum=0.9, weight_decay=5e-4)


def train():
    best_train_acc = 0
    best_validate_acc = 0
    for epoch in range(1, conf["parameters"]["epoch_size"]+1):
        print(f"\n>> Epoch: {epoch}")
        print("## train")
        net.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_acc = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            loss = train_loss / (batch_idx + 1)
            train_acc = 100. * train_correct / train_total
            progress_bar(batch_idx, len(train_loader),
                         f"Loss: {loss:.3f} | Acc: {train_acc:.3f} | {train_correct}/{train_total}")

        print("## validate")
        validate_correct = 0
        validate_total = 0
        validate_acc = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validate_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                validate_total += targets.size(0)
                validate_correct += predicted.eq(targets).sum().item()

                validate_acc = 100. * validate_correct / validate_total
                progress_bar(batch_idx, len(validate_loader), f"Acc: {validate_acc:.3f}")

        if train_acc > best_train_acc and validate_acc < best_validate_acc:
            break
        else:
            if train_acc > best_train_acc:
                best_train_acc = train_acc

            if validate_acc > best_validate_acc:
                best_validate_acc = validate_acc

        if train_acc > 90 or validate_acc > 90:
            torch.save(net.state_dict(), f"./save/epoch_{epoch}-train_acc_{train_acc}-validate_acc_{validate_acc}.ckpt")


if __name__ == "__main__":
    train()
