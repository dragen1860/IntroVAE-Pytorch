import  torch
import  numpy as np
from    torch.utils.data import DataLoader
import  argparse

from    celeba import load_celeba, unnorm_
from    model import IntroVAE
import  visdom













def main(args):
    print(args)

    torch.manual_seed(22)
    np.random.seed(22)

    viz = visdom.Visdom()

    db = load_celeba(args.root, args.imgsz)
    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda')
    vae = IntroVAE(args).to(device)
    params = filter(lambda x: x.requires_grad, vae.parameters())
    num = sum(map(lambda x: np.prod(x.shape), params))
    print('Total trainable tensors:', num)
    print(vae)

    viz.line([0], [0], win='encoder_loss', opts=dict(title='encoder_loss'))
    viz.line([0], [0], win='decoder_loss', opts=dict(title='decoder_loss'))
    viz.line([0], [0], win='ae_loss', opts=dict(title='ae_loss'))
    viz.line([0], [0], win='reg_ae', opts=dict(title='reg_ae'))
    viz.line([0], [0], win='encoder_adv', opts=dict(title='encoder_adv'))
    viz.line([0], [0], win='decoder_adv', opts=dict(title='decoder_adv'))


    # pre-training
    print('>>pre-training...')
    epoch_start = 0
    vae.set_alph_beta_gamma(0, args.beta, 0)
    for _ in range(2):

        db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)

        for _, (x, label) in enumerate(db_loader):
            x = x.to(device)
            epoch_start += 1

            encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp = vae(x)

            if epoch_start % 15 == 0:

                print(epoch_start, encoder_loss.item(), decoder_loss.item(), loss_ae.item())

                viz.line([encoder_loss.item()], [epoch_start], win='encoder_loss', update='append')
                viz.line([decoder_loss.item()], [epoch_start], win='decoder_loss', update='append')
                viz.line([loss_ae.item()], [epoch_start], win='ae_loss', update='append')
                viz.line([reg_ae.item()], [epoch_start], win='reg_ae', update='append')
                viz.line([encoder_adv.item()], [epoch_start], win='encoder_adv', update='append')
                viz.line([decoder_adv.item()], [epoch_start], win='decoder_adv', update='append')

            if epoch_start % 100 == 0:
                x, xr, xp = x[:8], xr[:8], xp[:8]
                viz.histogram(xr[0].view(-1), win='xr_hist', opts=dict(title='xr_hist'))
                unnorm_(x, xr, xp)
                viz.images(x, nrow=4, win='x', opts=dict(title='x'))
                viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
                viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))




    # training.
    print('>>training Intro-VAE now...')
    vae.set_alph_beta_gamma(args.alpha, args.beta, args.gamma)
    for epoch in range(epoch_start, args.epoch):

        try:
            x, label = iter(db_loader).next()
        except StopIteration as err:
            db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
            x, label = iter(db_loader).next()
            print('>>>iter over once.')

        x = x.to(device)

        encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp = vae(x)

        if epoch % 10 == 0:

            print(epoch, encoder_loss.item(), decoder_loss.item(), loss_ae.item())

            viz.line([encoder_loss.item()], [epoch], win='encoder_loss', update='append')
            viz.line([decoder_loss.item()], [epoch], win='decoder_loss', update='append')
            viz.line([loss_ae.item()], [epoch], win='ae_loss', update='append')
            viz.line([reg_ae.item()], [epoch], win='reg_ae', update='append')
            viz.line([encoder_adv.item()], [epoch], win='encoder_adv', update='append')
            viz.line([decoder_adv.item()], [epoch], win='decoder_adv', update='append')

        if epoch % 50 == 0:
            x, xr, xp = x[:8], xr[:8], xp[:8]
            viz.histogram(xr[0].view(-1), win='xr_hist', opts=dict(title='xr_hist'))
            unnorm_(x, xr, xp)
            viz.images(x, nrow=4, win='x', opts=dict(title='x'))
            viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
            viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))











if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=18, help='batch size')
    argparser.add_argument('--z_dim', type=int, default=256, help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=200000, help='epoches')
    argparser.add_argument('--margin', type=int, default=110, help='margin')
    argparser.add_argument('--alpha', type=float, default=0.25, help='alpha * loss_adv')
    argparser.add_argument('--beta', type=float, default=0.5, help='beta * ae_loss')
    argparser.add_argument('--gamma', type=float, default=1., help='gamma * kl(q||p)_loss')
    argparser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    argparser.add_argument('--root', type=str, default='/home/i/tmp/MAML-Pytorch/miniimagenet/', help='root/label/*.jpg')


    args = argparser.parse_args()

    main(args)