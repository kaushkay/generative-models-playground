from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size:int, train: bool=True):
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="../data", train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader


