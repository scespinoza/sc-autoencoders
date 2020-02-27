import argparse
import warnings
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from preprocess import GSE
from models import *

warnings.filterwarnings("ignore")

models_dict = {
    'stacked': AutoEncoder,
    'vae': VAE,
    'vade': VaDE,
    'zi_stacked': ZIAutoEncoder,
    'zi_vae': ZIVAE,
    'zi_vade': ZIVaDE
}

losses = {
    'stacked': losses.binary_crossentropy,
    'zi_stacked': losses.binary_crossentropy,
    'vae': lambda x, x_hat: 0.,
    'zi_vae': lambda x, x_hat: 0.,
    'vade': lambda x, x_hat: 0.,
    'zi_vade': lambda x, x_hat: 0.
}


def load_data(args):
        """
    Helper function to train a model.

    Parameters
    ----------
    args: Namespace,
        Arguments for loading data.
    """
    if args.model == 'vade' and args.dataset in ['GSE57872', 'GSE84465']:
        assert args.class_name != '', "Must provide a class name."

    dataset = GSE(name=args.dataset, class_name=args.class_name)

    name = args.dataset + '_' + args.class_name + '_' + args.model
    
    if args.split:
        x_train, y_train = dataset.train
        x_test, y_test = dataset.test

    else:
        x_train = dataset.data_scaled
        x_test = dataset.data_scaled
        y_train = dataset.class_labels
        y_test = dataset.class_labels

    if args.components == 0:
        n_components = len(np.unique(dataset.class_labels))
    else:
        n_components = args.components

    return (x_train, x_test), (y_train, y_test), n_components, dataset

def train_model(args):
    """
    Helper function to train a model.

    Parameters
    ----------
    args: Namespace,
        Arguments for training model.
    """
    name = args.dataset + '_' + args.class_name + '_' + args.model
    x, y, n_components, dataset = load_data(args)

    if 'vade' in args.model:
        model = models_dict[args.model](original_dim=dataset.n_genes, 
                                        latent_dim=args.latent, 
                                        n_components=n_components,
                                        pretrain=args.pretrain,
                                        pretrain_lr=args.pretrain_lr,
                                        k=args.k,
                                        search_k = args.search_k,
                                        name=name)
    else:
        model = models_dict[args.model](original_dim=dataset.n_genes,
                                        latent_dim=args.latent,
                                        name=name)


    optimizer = optimizers.Adam(learning_rate=args.lr)

    def scheduler(epoch):
        # learning rate scheduler
        return args.lr * (args.decay ** (epoch // args.lr_interval))

    # callbacks
    early_stopping = callbacks.EarlyStopping(patience=args.patience)
    model_checkpoint = callbacks.ModelCheckpoint('weights/' + name + '_trained.h5',
                                                 save_best_only=True,
                                                 save_weights_only=True)
    lr_scheduler = callbacks.LearningRateScheduler(scheduler)

    plot_latent = PlotLatentSpace(dataset.data_scaled, dataset.class_labels, interval=args.interval)
    
    model.compile(optimizer=optimizer, loss=losses[args.model])
    
    callbacks_list = [early_stopping, model_checkpoint, plot_latent, lr_scheduler]

    if 'vade' in args.model:
        accuracy = PrintLossAndAccuracy(dataset.data_scaled, dataset.class_labels)        
        callbacks_list += [accuracy]
        if args.warmup:
            callbacks_list += [WarmUpCallback(k=args.k)]
    if 'zi' in args.model:
        annealing = TauAnnealing(gamma=3e-4)
        callbacks_list += [annealing]
        

    print("Training model: " + name)
    history = model.fit(x[0], x[0], epochs=args.epochs, validation_data=(x[1], x[1]),
                callbacks=callbacks_list, verbose=args.verbose, batch_size=args.batch_size)
    return history, name


def plot_output(args, history, name):
    """
    Helper function to plot training outputs.

    Parameters
    ----------
    args: Namespace,
        Contains the training parameters of the model.
    history: dict,
        Dictionary containing the training history.
    name: str,
        Model name.
    """
    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv('results/' + name + '_history.csv', index=False, sep='\t')
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.title('Learning Curves')
    plt.savefig('figures/' + name + '/history.png')
    plt.close()

    if 'vade' in args.model:
        plt.plot(history.history['test_metric'])
        plt.title('Accuracy')
        plt.savefig('figures/' + name + '/accuracy.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GSE57872', help='dataset to train')
    parser.add_argument('--split', action='store_true', help='split dataset on train/test')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent', type=int, default=10, help='latent dimensions')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
    parser.add_argument('--interval', type=int, default=20, help='interval (epochs) to plot latent space')
    parser.add_argument('--model', type=str, default='vade', help='model to train')
    parser.add_argument('--pretrain', type=int, default=0, help='pretrain vade')
    parser.add_argument('--pretrain_lr', type=float, default=3e-4, help='pretrain_lr')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--components', type=int, default=0, help='GMM components')
    parser.add_argument('--lr_interval', type=int, default=10, help='interval for lr update')
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--class_name', type=str, default='', help='class to do clustering. only for datasets GSE84465 and GSE57872')
    parser.add_argument('--k', type=float, default=1., help='initial contribution of vade loss')
    parser.add_argument('--warmup', action='store_true', help='warmup for kappa parameter')
    parser.add_argument('--search_k', action='store_true', help='search for best k in the latent space')

    args = parser.parse_args()

    if args.dataset in ['GSE57872', 'GSE84465']:
        assert args.class_name != '', "Must provide a class name."

    if args.model == 'all':
        for model in models_dict.keys():
            args.model = model
            history, name = train_model(args)
            plot_output(args, history, name)
    else:
        history, name = train_model(args)
        plot_output(args, history, name)
