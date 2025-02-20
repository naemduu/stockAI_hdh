import torch
import time
import numpy as np
import torch.optim as optim
from learning.load_data import create_data_loaders
from models import DNNModel, StockTransformer


def train_epoch(args, epoch, model, data_loader, optimizer):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, batch in enumerate(data_loader):
        inputs, labels = batch
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        price, sigma = model(inputs)
        # loss = (price - labels)**2 / sigma + sigma
        loss = torch.abs(price - labels)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    start = time.perf_counter()

    metric_loss = 0
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            inputs, labels = batch
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            price, sigma = model(inputs)
            metric_loss = metric_loss + torch.sum(torch.abs(price-labels)/torch.abs(labels))
    metric_loss = metric_loss.detach().cpu()
    num_subjects = len(data_loader)
    metric_loss = metric_loss / num_subjects
    return metric_loss, num_subjects, time.perf_counter() - start


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = StockTransformer(input_dim=20)
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path=args.data_path, args=args, shuffle=True)
    val_loader = create_data_loaders(data_path=args.data_path, args=args, eval=True)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer)
        val_loss, num_subjects, val_time = validate(args, model, val_loader)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        #save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #start = time.perf_counter()
            #save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            #print(
            #    f'ForwardTime = {time.perf_counter() - start:.4f}s',
            #)
