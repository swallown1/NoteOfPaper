from torch import optim
import time
from transformer import Transformer
from load_data import *
import torch
from layers import label_smoothing




if __name__ == '__main__':
    print_batchs = 10
    device = torch.device('cuda')
    data_sour = Data_source()

    train_data = My_dataset(data_sour)
    train_loader = data.DataLoader(train_data, batch_size=arg.batch_size, shuffle=False)

    model = Transformer(data_sour.num_words_en,data_sour.num_words_de,arg.hidden_units).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    loss_fn= torch.nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(arg.num_epochs):
        for t, (x, y)in enumerate(train_loader):
            x=x.to(device=torch.device('cuda'))
            y=y.to(device=torch.device('cuda'))
            dec_ipt = torch.cat((torch.ones_like(y[:,:1])*2,y[:,:-1]),-1)
            logits,preds = model(x,dec_ipt)

            istargets = torch.le(preds, 0).float()
            # acc = torch.sum((torch.ne(preds, y).float() * istargets )/ torch.sum(istargets))
            # print(type(y))

            # y_smoothed = label_smoothing(torch.nn.functional.one_hot(y.type_as(torch.tensor(1)), num_classes=data_sour.num_words_en))
            logits=logits.view(10*32,-1)
            y = y.view(10 * 32, -1).long()
            loss = loss_fn(logits,y)
            mean_loss = torch.sum(loss * istargets) / (torch.sum(istargets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if  t % print_batchs == 0:
                acc = torch.sum((torch.ne(preds, y).float() * istargets) / torch.sum(istargets))
                print('Epoch %d batch %d time %d, loss = %.4f, acc = %.4f' % (epoch, t, time.time() - t, loss.item(),acc))




