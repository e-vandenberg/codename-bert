import torch
import random
from guesser import SiameseBert
import sys

if __name__ == '__main__':

    net = SiameseBert()
    # read training data, split it up to avoid taking too long to train
    data_split = 10
    count = 0
    with open("data/labelled_data.csv", "r") as datafile:
        for line in datafile.readlines():
            if count % data_split == 0:
                row = line.strip().split(",")
                row[-1] = int(row[-1])
                data.append(row)
            count += 1

    train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)
    loss_fn = torch.nn.BCELoss()
    epochs = 25

    for epoch in range(epochs):
        counter = 0
        loss_val = 0
        for word1, word2, label in train_loader:
            sys.stdout.write('\r')
            optimizer.zero_grad()

            # get output, adjust
            output = net(word1[0], word2[0])
            loss = loss_fn(output, label.float())
            loss_val += loss.item()
            loss.backward()
            optimizer.step()
            counter += 1

            # continuous updates without clutter
            sys.stdout.write(f"--> ({epoch + 1}/{epochs}): {counter}/{len(data)} ")
            sys.stdout.flush()

        # print epoch loss
        print("loss:", loss_val)

    # save model once training is complete
    torch.save(net.state_dict(), 'saved_model.pt')
