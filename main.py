import  os, glob
import  torch
import  numpy as np
from    torch.utils.data import DataLoader
import  argparse
from    torchvision.utils import save_image

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
    # print(vae)

    for path in ['res', 'ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)


    epoch_start = 0
    if args.resume is not None and args.resume != 'None':
        if args.resume is '': # load latest
            ckpts = glob.glob('ckpt/*_*.mdl')
            if not ckpts:
                print('no avaliable ckpt found.')
                raise FileNotFoundError
            ckpts = sorted(ckpts, key=os.path.getmtime)
            # print(ckpts)
            ckpt = ckpts[-1]
            epoch_start = int(ckpt.split('.')[-2].split('_')[-1])
            vae.load_state_dict(torch.load(ckpt))
            print('load latest ckpt from:', ckpt, epoch_start)
        else: # load specific ckpt
            if os.path.isfile(args.resume):
                vae.load_state_dict(torch.load(args.resume))
                print('load ckpt from:', args.resume, epoch_start)
            else:
                raise FileNotFoundError
    else:
        print('pre-training and training from scratch...')

    viz.line([[0 for _ in range(6)]], [epoch_start], win='train', opts=dict(title='training',
                    legend=['b*ae', 'a*inf(x)', 'a*inf(xr)', 'a*inf(xp)', 'a*gen(xr)', 'a*gen(xp)']))
    viz.line([0], [epoch_start], win='encoder_loss', opts=dict(title='encoder_loss'))
    viz.line([0], [epoch_start], win='decoder_loss', opts=dict(title='decoder_loss'))
    viz.line([0], [epoch_start], win='ae_loss', opts=dict(title='ae_loss'))
    viz.line([0], [epoch_start], win='reg_ae', opts=dict(title='reg_ae'))
    viz.line([0], [epoch_start], win='encoder_adv', opts=dict(title='encoder_adv'))
    viz.line([0], [epoch_start], win='decoder_adv', opts=dict(title='decoder_adv'))


    # pre-training for 1.5 epoch
    # pretraining_epoch = 1.5 * training_splitting_size // batchsz
    pretraining_epoch = int(1. * (0.9*len(db)) ) // args.batchsz
    if pretraining_epoch - epoch_start > 0:
        print('>>pre-training for %d epoches,'%pretraining_epoch, 'already completed:', epoch_start)
        vae.set_alph_beta_gamma(0, args.beta, args.gamma)

        # pre-training for at most 2 iteration.
        for _ in range(2):
            db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
            print('epoch\tvae\tenc-adv\t\tdec-adv\t\tae\t\tenc\t\tdec')

            for _, (x, label) in enumerate(db_loader):
                x = x.to(device)

                encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp, \
                    regr, regr_ng, regpp, regpp_ng = vae(x)

                if epoch_start % 50 == 0:

                    print(epoch_start, '\t%0.3f\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f'%(
                        reg_ae.item(), encoder_adv.item(), decoder_adv.item(), loss_ae.item(), encoder_loss.item(),
                        decoder_loss.item()
                    ))

                    viz.line([[args.beta*loss_ae.item(), args.gamma*reg_ae.item(), args.alpha*regr_ng.item(),
                              args.alpha * regpp_ng.item(), args.alpha*regr.item(), args.alpha*regpp.item()]],
                             [epoch_start], win='train', update='append')
                    viz.line([encoder_loss.item()], [epoch_start], win='encoder_loss', update='append')
                    viz.line([decoder_loss.item()], [epoch_start], win='decoder_loss', update='append')
                    viz.line([loss_ae.item()], [epoch_start], win='ae_loss', update='append')
                    viz.line([reg_ae.item()], [epoch_start], win='reg_ae', update='append')
                    viz.line([encoder_adv.item()], [epoch_start], win='encoder_adv', update='append')
                    viz.line([decoder_adv.item()], [epoch_start], win='decoder_adv', update='append')

                if epoch_start % 200 == 0:
                    x, xr, xp = x[:8], xr[:8], xp[:8]
                    viz.histogram(xr[0].view(-1), win='xr_hist', opts=dict(title='xr_hist'))
                    unnorm_(x, xr, xp)
                    viz.images(x, nrow=4, win='x', opts=dict(title='x'))
                    viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
                    viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))

                    if epoch_start % 10000 == 0:
                        save_image(xr, 'res/xr_%d.jpg'%epoch_start, nrow=4)
                        save_image(xp, 'res/xp_%d.jpg'%epoch_start, nrow=4)
                        print('save xr xp to res directory.')

                if epoch_start % 10000 == 0:
                    torch.save(vae.state_dict(), 'ckpt/introvae_%d.mdl'%epoch_start)
                    print('saved ckpt:', 'ckpt/introvae_%d.mdl'%epoch_start)

                epoch_start += 1
                if epoch_start > pretraining_epoch:
                    break




    # training.
    print('>>training Intro-VAE now...')
    vae.set_alph_beta_gamma(args.alpha, args.beta, args.gamma)
    db_iter = iter(db_loader)
    print('epoch\tvae\tenc-adv\t\tdec-adv\t\tae\t\tenc\t\tdec')
    for epoch in range(epoch_start, args.epoch):

        try:
            # can not use iter(db_loader).next()
            x, label = next(db_iter)
        except StopIteration as err:
            db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
            db_iter = iter(db_loader)
            x, label = next(db_iter)
            print('epoch\tvae\tenc-adv\t\tdec-adv\t\tae\t\tenc\t\tdec')

        x = x.to(device)

        encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp, \
                regr, regr_ng, regpp, regpp_ng = vae(x)

        if epoch % 100 == 0:

            print(epoch_start, '\t%0.3f\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f\t\t%0.3f' % (
                reg_ae.item(), encoder_adv.item(), decoder_adv.item(), loss_ae.item(), encoder_loss.item(),
                decoder_loss.item()
            ))

            viz.line([[args.beta * loss_ae.item(), args.gamma * reg_ae.item(), args.alpha * regr_ng.item(),
                      args.alpha * regpp_ng.item(), args.alpha * regr.item(), args.alpha * regpp.item()]],
                     [epoch], win='train', update='append')
            viz.line([encoder_loss.item()], [epoch], win='encoder_loss', update='append')
            viz.line([decoder_loss.item()], [epoch], win='decoder_loss', update='append')
            viz.line([loss_ae.item()], [epoch], win='ae_loss', update='append')
            viz.line([reg_ae.item()], [epoch], win='reg_ae', update='append')
            viz.line([encoder_adv.item()], [epoch], win='encoder_adv', update='append')
            viz.line([decoder_adv.item()], [epoch], win='decoder_adv', update='append')

        if epoch % 500 == 0:
            x, xr, xp = x[:8], xr[:8], xp[:8]
            viz.histogram(xr[0].view(-1), win='xr_hist', opts=dict(title='xr_hist'))
            unnorm_(x, xr, xp)
            viz.images(x, nrow=4, win='x', opts=dict(title='x'))
            viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
            viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))

            if epoch % 10000 == 0:
                save_image(xr, 'res/xr_%d.jpg' % epoch, nrow=4)
                save_image(xp, 'res/xp_%d.jpg' % epoch, nrow=4)
                print('save xr, xp to res directory')

        if epoch % 10000 == 0:
            torch.save(vae.state_dict(), 'ckpt/introvae_%d.mdl'%epoch)
            print('saved ckpt:', 'ckpt/introvae_%d.mdl'%epoch)




    torch.save(vae.state_dict(), 'ckpt/introvae_%d.mdl'%args.epoch)
    print('saved final ckpt:', 'ckpt/introvae_%d.mdl'%args.epoch)






if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=8, help='batch size')
    argparser.add_argument('--z_dim', type=int, default=256, help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=750000, help='epoches')
    argparser.add_argument('--margin', type=int, default=110, help='margin')
    argparser.add_argument('--alpha', type=float, default=0.25, help='alpha * loss_adv')
    argparser.add_argument('--beta', type=float, default=0.5, help='beta * ae_loss')
    argparser.add_argument('--gamma', type=float, default=1., help='gamma * kl(q||p)_loss')
    argparser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argparser.add_argument('--root', type=str, default='/home/i/dbs/',
                           help='root/label/*.jpg')
    argparser.add_argument('--resume', type=str, default=None,
                           help='with ckpt path, set None to train from scratch, set empty str to load latest ckpt')


    args = argparser.parse_args()

    main(args)