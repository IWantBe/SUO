import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
import models
import argparse
import json
from typing import Dict


def get_params_num(model: nn.Module) -> int:
    '''Calculate the number of model parameters'''
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def train(args, src, tar):
    print(f'{args.dataset}: {src} -> {tar}')

    # read data
    rate = [1.0, 0, 0]
    rate_target = [0.9, 0, 0.1]
    if args.dataset == 'cwru':
        get_data = datasets.get_data_cwru
        sample_len = 420
        stride = sample_len // 2
        num_classes = 10
    elif args.dataset == 'jnu':
        get_data = datasets.get_data_jnu
        sample_len = 500
        stride = sample_len // 2
        num_classes = 4
    elif args.dataset == 'pu':
        get_data = datasets.get_data_pu
        sample_len = 430
        stride = sample_len // 2
        num_classes = 3

    train_data_source, train_data_target, test_data_target = get_data(src, tar, rate, rate_target, stride, sample_len)
    train_dataloader_source = DataLoader(train_data_source, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_dataloader_target = DataLoader(train_data_target, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader_target = DataLoader(test_data_target, batch_size=1, shuffle=True, drop_last=True)

    # model
    MODEL = eval(f'models.{args.model}')
    model = MODEL(sample_len, num_classes, **args.model_config).to(args.device)
    print(f'model params num: {get_params_num(model)}')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for e in range(args.epoch):

        model.train()
        iter_source = iter(train_dataloader_source)
        iter_target = iter(train_dataloader_target)
        for _ in range(len(train_dataloader_source)):
            try:
                data_source, label_source = next(iter_source)
            except StopIteration:
                iter_source = iter(train_dataloader_source)
                data_source, label_source = next(iter_source)

            try:
                data_target, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(train_dataloader_target)
                data_target, _ = next(iter_target)

            data_source = data_source.to(args.device)
            label_source = label_source.to(args.device)
            data_target = data_target.to(args.device)

            loss = model(data_source, data_target, label_source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_dataloader_target:
                data, target = data.to(args.device), target.to(args.device)
                pred = model.predict(data)
                test_loss += F.cross_entropy(pred, target).item()
                correct += (pred.argmax(1) == target).sum().item()

        test_loss /= len(test_dataloader_target)
        acc = correct / len(test_dataloader_target.dataset)
        print(f'Epoch: {e+1}/{args.epoch}, '
              f'Average loss: {test_loss:.5f}, '
              f'Accuracy: {correct}/{len(test_dataloader_target.dataset)} ({100*acc:.2f}%)')

    print()
    return acc, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--trials', '-t', type=int, default=5)
    parser.add_argument('--epoch', '-e', type=int, default=80)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--dataset', '-ds', type=str, default='cwru')

    parser.add_argument('--source', '-src', nargs='+', type=int, default=[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3])
    parser.add_argument('--target', '-tar', nargs='+', type=int, default=[1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2])

    parser.add_argument('--model', '-m', type=str, default='SUO')
    parser.add_argument('--model_config', '-mc', type=json.loads, default='{}')

    parser.add_argument('--save', type=str, default='')

    args = parser.parse_args()

    result: Dict[str, list] = {}
    src = args.source
    tar = args.target
    for trial in range(args.trials):
        torch.cuda.empty_cache()
        print(f"第{trial+1}/{args.trials}次实验")

        for i in range(len(src)):
            s = f'{src[i]}->{tar[i]}'
            if s not in result: result[s] = []
            acc, model = train(args, src[i], tar[i])
            result[s].append(acc)
            if args.save: torch.save(model.state_dict(), args.save)

    print()
    for k, v in result.items():
        print(f'{k}: ', end='')
        for x in v:
            print(f'{x:.4f} ', end='')
        print()
