import argparse
import warnings
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from preprocess import GSE
from models import *

warnings.filterwarnings("ignore")

models_dict = {
    'stacked': AutoEncoder,
    'vae': VariationalAutoEncoder,
    'vade': VariationalDeepEmbedding
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GSE57872', help='dataset to train')
    parser.add_argument('--split', action='store_true', help='split dataset on train/test')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--latent', type=int, default=10, help='latent dimensions')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping')
    parser.add_argument('--interval', type=int, default=20, help='interval (epochs) to plot latent space')
    parser.add_argument('--model', type=str, default='vade', help='model to train')
    parser.add_argument('--pretrain', type=int, default=0, help='pretrain vade')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--components', type=int, default=0, help='GMM components')
    parser.add_argument('--lr_interval', type=int, default=10, help='interval for lr update')

    args = parser.parse_args()

    dataset = GSE(name=args.dataset)

    name = args.dataset + '_' + args.model
    
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

    
    if args.model == 'vade':
        model = models_dict[args.model](original_dim=dataset.n_genes, 
                                        latent_dim=args.latent, 
                                        n_components=n_components,
                                        pretrain=args.pretrain,
                                        name=name)
    else:
        model = models_dict[args.model](original_dim=dataset.n_genes,
                                        latent_dim=args.latent,
                                        name=name)
    optimizer = optimizers.Adam(learning_rate=args.lr)
    loss = lambda x, x_hat: 0.


    # callbacks
    early_stopping = callbacks.EarlyStopping(patience=args.patience)
    model_checkpoint = callbacks.ModelCheckpoint('weights/' + name + '_trained.h5',
                                                 save_best_only=True,
                                                 save_weights_only=True)

    
    def scheduler(epoch):
        # learning rate scheduler
        return args.lr * (0.9 ** (epoch // args.lr_interval))

    lr_scheduler = callbacks.LearningRateScheduler(scheduler)
    accuracy = PrintLossAndAccuracy(model, dataset.data_scaled, dataset.class_labels)
    plot_latent = PlotLatentSpace(model, dataset.data_scaled, dataset.classgit_labels, interval=args.interval)
    model.compile(optimizer=optimizer, loss=loss)

    print("Training model: " + name)
    history = model.fit(x_train, x_train, epochs=args.epochs, validation_data=(x_test, x_test),
                callbacks=[early_stopping, lr_scheduler, accuracy, plot_latent, model_checkpoint], verbose=args.verbose)

    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv('results/' + name + '_history.csv', index=False, sep='\t')
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.title('Learning Curves')
    plt.savefig('figures/' + name + '/history.png')
    plt.close()

    plt.plot(history.history['test_metric'])
    plt.title('Accuracy')
    plt.savefig('figures/' + name + '/accuracy.png')