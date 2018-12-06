import  torch
import  numpy as np
from    torch.utils.data import DataLoader
import  argparse

from    celeba import load_celeba, unnorm_
from    model import IntroVAE
import  visdom













def main(args):

    torch.manual_seed(22)
    np.random.seed(22)

    viz = visdom.Visdom()

    db = load_celeba('/home/i/tmp/MAML-Pytorch/miniimagenet/', args.imgsz)
    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda')
    vae = IntroVAE(args.imgsz, args.z_dim).to(device)
    params = filter(lambda x: x.requires_grad, vae.parameters())
    num = sum(map(lambda x: np.prod(x.shape), params))
    print('Total trainable tensors:', num)
    # print(vae)

    viz.line([0], [0], win='encoder_loss', opts=dict(title='encoder_loss'))
    viz.line([0], [0], win='decoder_loss', opts=dict(title='decoder_loss'))
    viz.line([0], [0], win='ae_loss', opts=dict(title='ae_loss'))

    for epoch in range(args.epoch):

        try:
            x, label = iter(db_loader).next()
        except Exception as err:
            db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
            x, label = iter(db_loader).next()
            print('iter over once.')

        x = x.to(device)
        # print(x.shape, label)
        # print(db.class_to_idx)

        encoder_loss, decoder_loss, loss_ae, xr, xp = vae(x)

        if epoch % 10 == 0:
            viz.line([encoder_loss.item()], [epoch], win='encoder_loss', update='append')
            viz.line([decoder_loss.item()], [epoch], win='decoder_loss', update='append')
            viz.line([loss_ae.item()], [epoch], win='loss_ae', update='append')

            if epoch % 50 == 0:
                x, xr, xp = x[:8], xr[:8], xp[:8]
                unnorm_(x, xr, xp)
                viz.images(x, nrow=4, win='x', opts=dict(title='x'))
                viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
                viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))











if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=32, help='batch size')
    argparser.add_argument('--z_dim', type=int, default=64, help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=200000, help='epoches')
    args = argparser.parse_args()

    main(args)