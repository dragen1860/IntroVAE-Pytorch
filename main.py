import  torch
import  numpy as np
from    torch.utils.data import DataLoader


from    celeba import load_celeba
from    model import IntroVAE















def main():

    torch.manual_seed(22)
    np.random.seed(22)

    db = load_celeba('/home/i/tmp/MAML-Pytorch/miniimagenet/')
    db_loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda')
    vae = IntroVAE(512).to(device)
    print(vae)

    for epoch, (x, label) in enumerate(db_loader):

        x = x.to(device)
        # print(x.shape, label)
        # print(db.class_to_idx)

        vae(x)









if __name__ == '__main__':
    main()