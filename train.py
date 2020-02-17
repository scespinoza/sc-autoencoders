
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

from rna_autoencoder import AutoEncoder, random_projection, plot_dimensionality_reduction


def cv_autoencoder(data, epochs=2000):
    _, input_dim = data.shape
    X_train, X_test = train_test_split(data.values)

    estimator = KerasRegressor(build_fn=AutoEncoder.build_autoencoder, name='all_data', input_dim=input_dim)

    # callbacks
    early_stopping = EarlyStopping(patience=10)
    tensorboard = TensorBoard('tmp/all_data')

    # grid
    param_dist = {
        'learning_rate': [1e-4, 5e-4, 1e-5, 5e-5],
        'batch_size': [16, 32],
    }

    # cross-validate
    rand_search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=param_dist,
                                    n_iter=5,
                                    cv=3,
                                    refit=True,
                                    scoring='neg_mean_absolute_error')

    rand_search.fit(X_train, X_train, 
                    validation_data=(X_test, X_test), 
                    callbacks=[early_stopping, tensorboard],
                    epochs=epochs)
    return rand_search


if __name__ == '__main__':

    data = pd.read_csv('clean_data/all_data.txt', sep='\t', index_col=0).T
    rand_search = cv_autoencoder(data, epochs=500)
    model = rand_search.best_estimator_.model
    filepath = model.save_model()
    model.plot_history()
    plt.savefig(filepath + '/history.png')
    cv_results = pd.DataFrame(rand_search.cv_results_)
    cv_results.to_csv(filepath + '/cv-results.csv', index=False)



