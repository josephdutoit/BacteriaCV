import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

import gc
from tqdm import tqdm

from bacteria_dataset import BacteriaDataset
from bacteria_model import BacteriaModel

def accuracy(y_hat, y_truth):
    # Basic accuracy function
    dist = nn.PairwiseDistance(y_hat, y_truth)
    acc = torch.mean(dist < 10).item()
    return acc

def evaluate(model, objective, val_loader, device, epoch):
    val_losses = 0
    val_accs = 0
    batches = 0
    model.eval()
    # preds = []
    # top5_preds = []
    with torch.no_grad():
        for x, y_truth in val_loader:

            batches += 1

            # Get validation loss and predictions
            x, y_truth = x.to(device), y_truth.to(device)
            y_hat = model(x)
            val_loss = objective(y_hat, y_truth)
            val_acc = accuracy(y_hat, y_truth)
            # preds.append(LABEL_LIST[int(torch.argmax(y_hat, dim=1).item())])

            # Do top 5
            # _, ind = y_hat.topk(5, dim=1, largest=True)
            # ind = ind.tolist()[0]
            # top5_preds.append([LABEL_LIST[int(ind[i])] for i in range(len(ind))])

            val_losses += val_loss.item()
            val_accs += val_acc

    # Write the top1 and top5 predictions to file
    # print("Writing to /home/jcdutoit/Snackathon/val_" + str(epoch) + ".txt'")
    # with open('/home/jcdutoit/Snackathon/val_'+str(epoch)+'.txt', 'w') as f:
    #     for pred in preds:
    #         f.write(pred + '\n')

    # with open('/home/jcdutoit/Snackathon/top5_val_'+str(epoch)+'.txt', 'w') as f:
    #     for idx in top5_preds:
    #         pred_string = ', '.join(idx)
    #         f.write(pred_string + '\n')

    model.train()

    return val_losses/batches, val_accs/batches

def train(start_frozen=False, model_unfreeze=0):
    """Fine-tunes a CNN
    Args:
        start_frozen (bool): whether to start with the network weights frozen.
        model_unfreeze (int): the maximum number of network layers to unfreeze
    """

    gc.collect()
    epochs = 5
    # Start with a very low learning rate
    lr = .00005
    val_every = 500
    num_classes = 2
    batch_size = 32
    data_length = 2770
    device = torch.device('cuda:0')

    # Initialize datasets and dataloaders
    train_dataset, val_dataset = random_split(BacteriaDataset(), [int(data_length * 0.8), int(data_length * 0.2)])

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=batch_size)
    val_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers=8,
                              batch_size=1)

    # Model
    model = BacteriaModel(num_classes, start_frozen=start_frozen).to(device)

    # Objective
    objective = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)

    # Progress bar
    pbar = tqdm(total=len(train_loader) * epochs)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Main training loop
    cnt = 0
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        if cnt < model_unfreeze and cnt % 45 == 0:
            # Try unfreezing. It didn't work so well for us.
            layers = int(cnt / 45)
            print("\nUnfreezing " + str(layers) + " layers")
            model.unfreeze(layers)

        for x, y_truth in train_loader:
            
            x, y_truth = x.to(device), y_truth.to(device)

            optimizer.zero_grad()

            # Training
            y_hat = model(x)
            train_loss = objective(y_hat, y_truth)
            train_acc = accuracy(y_hat, y_truth)

            train_loss.backward()
            optimizer.step()

            train_accs.append(train_acc)
            train_losses.append(train_loss.item())

            # Validation
            if cnt % val_every == 0:
                val_loss, val_acc = evaluate(model, objective, val_loader, device, cnt)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                # torch.save(model, '/home/jcdutoit/Snackathon/' + str(epoch) + '_snack_model.pt')

            # Update progress bar
            pbar.set_description('train loss:{:.4f}, train accuracy:{:.4f}, val loss:{:.4f}, val accuracy:{:.4f}.'.format(train_loss.item(), train_acc, val_losses[-1], val_accs[-1]))
            pbar.update(1)
            cnt += 1

    pbar.close()

if __name__ == "__main__":
    # Train with no unfreezing
    train(start_frozen=False, model_unfreeze=0)