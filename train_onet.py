import nets
import train

if __name__ == '__main__':
    net = nets.ONet()
    net.train()
    trainer = train.Trainer(net, './param/onet.pt', r"C:\celeba4\48", isTrainLandmark=True)
    trainer.train()
