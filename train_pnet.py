import nets
import train

if __name__ == '__main__':
    net = nets.PNet()

    trainer = train.Trainer(net, './param/pnet.pt', r"../img_celeba_4dbg/12")
    trainer.train()
