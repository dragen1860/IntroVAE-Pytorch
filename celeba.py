import  torch
from    torchvision import datasets, transforms


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        with torch.no_grad():

            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor[:, i,...].mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor

def load_celeba(root, imgsz):

    transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize([imgsz, imgsz]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    db = datasets.ImageFolder(root, transform=transform)

    return db


def unnorm_(*args):
    """
    conduct reverse normalize on each tensor in-place
    :param args:
    :return:
    """
    net = UnNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    for img in args:
        net(img)
