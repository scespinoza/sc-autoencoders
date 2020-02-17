import argparse
import torch 
import torch.utils.data
import pandas as pd
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from train import TrainerVaDE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000,
                        help="number of iterations")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='learning rate')
    parser.add_argument('--pretrained_path', type=str, default='weights/pretrained_parameter.pth',
                        help='Output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = pd.read_csv('data/biase/biase.txt', sep='\t', index_col=0).T

    X_scaled = MinMaxScaler().fit_transform(data.values)
    X_train, X_test = train_test_split(X_scaled)
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    vade = TrainerVaDE(args, device, dataloader)
    if args.pretrain==True:
        vade.pretrain()
    vade.train()

