import  torch
import  numpy as np
from    torch.utils.data import DataLoader
import  argparse

from    celeba import load_celeba
from    model import IntroVAE















def main(args):

    torch.manual_seed(22)
    np.random.seed(22)

    db = load_celeba('/home/i/tmp/MAML-Pytorch/miniimagenet/', args.imgsz)
    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda')
    vae = IntroVAE(args.imgsz, args.z_dim).to(device)
    params = filter(lambda x: x.requires_grad, vae.parameters())
    num = sum(map(lambda x: np.prod(x.shape), params))
    print('Total trainable tensors:', num)
    # print(vae)

    for epoch, (x, label) in enumerate(db_loader):

        x = x.to(device)
        # print(x.shape, label)
        # print(db.class_to_idx)

        vae(x)









if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=32, help='batch size')
    argparser.add_argument('--z_dim', type=int, default=64, help='hidden latent z dim')
    args = argparser.parse_args()

    main(args)