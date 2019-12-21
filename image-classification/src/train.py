import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import AlexNet

CLASS_NAMES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck'
)

BATCH_SIZE = 256
LR = 0.01
NUM_EPOCHS = 50

num_classes = len(CLASS_NAMES)


def epoch_train(train_loader, model, optimizer, criterion, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.train()
    model = model.to(device)
    epoch_loss = 0.
    total = 0
    correct = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(train_loader)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return epoch_loss, accuracy


def epoch_eval(eval_loader, model, criterion, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model = model.to(device)
    epoch_loss = 0.
    total = 0
    correct = 0
    for inputs, targets in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item() / len(eval_loader)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return epoch_loss, accuracy


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    model = AlexNet(num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = epoch_train(train_loader, model, optimizer,
                                            criterion)
        test_loss, test_acc = epoch_eval(test_loader, model, criterion)

        print(f'EPOCH: [{epoch}/{NUM_EPOCHS}]')
        print(f'TRAIN LOSS: {train_loss:.3f}, TRAIN ACC: {train_acc:.3f}')
        print(f'TEST LOSS: {test_loss:.3f}, TEST ACC: {test_acc:.3f}')

        parameters = model.state_dict()
        torch.save(parameters, f'../weights/{epoch}.pth')
