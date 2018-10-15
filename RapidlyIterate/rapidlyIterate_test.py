import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, Normalizer, PolynomialFeatures
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras import Sequential, optimizers
from keras.layers import InputLayer,Dense, BatchNormalization,Activation
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib.gridspec import  GridSpec

FOLDER_PATH = r'FILE PATH'
data = pd.read_csv(r'DATA PATH')
FEATURES = list(data)[:-1]
data = np.array(data)
M = data.shape[0]


def one_hot(data, save_path, name):
    labels = data[:, -1].reshape(data.shape[0], -1)
    data = data[:, :-1]
    nominal_cols = [1, 2, 3, 5, 6, 7, 8, 9, 10]  # columns that are categorical
    cont_cols = [x for x in range(data.shape[1]) if x not in nominal_cols]
    encoder = OneHotEncoder(sparse=False)
    list_of_one_hot_mats = []
    continuous_vars = np.zeros(shape=(data.shape[0], data.shape[1] - len(nominal_cols)))

    for col, new_col in zip(cont_cols, range(len(cont_cols))):
        continuous_vars[:, new_col] = data[:,col]

    for col in nominal_cols:
        min_val = min(data[:, col])
        if min_val < 0:
            data[:, col] += (-1 * min_val)
        feat_mat = encoder.fit_transform(data[:, col].reshape(data.shape[0], 1))
        list_of_one_hot_mats.append(feat_mat)
    nominal_features = np.concatenate(list_of_one_hot_mats,axis=1)
    new_data = np.concatenate((continuous_vars, nominal_features), axis=1)
    np.savetxt(save_path+name+'.csv', new_data, delimiter=',')

def preprocess_data(data, technique, labels=True):
    data_labels = None
    if labels:
        data_labels = data[:, -1]
        data = data[:, :-1]
        if technique == 'MinMax':
            transformed = MinMaxScaler().fit_transform(data)
        elif technique == 'Standard':
            transformed = StandardScaler().fit_transform(data)
        elif technique == "Robust":
            transformed = RobustScaler().fit_transform(data)
        elif technique == "MaxAbs":
            transformed = MaxAbsScaler().fit_transform(data)
        elif technique == "Quantile":
            transformed = QuantileTransformer().fit_transform(data)
        elif technique == "Normalizer":
            transformed = Normalizer().fit_transform(data)
        elif technique == "Polynomial":
            transformed = PolynomialFeatures().fit_transform(data)
        else:
            print("Not a valid preprocessing method for this function")
            return None

        if labels:
            return transformed, data_labels
        else:
            return transformed

def plot_3d(data, colors, title):
    x, y, z = data.T
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=colors)
    plt.title(title)
    plt.show()

def visualize_data(data):
    labels = data[:, -1]
    data = data[:, :-1]
    # --------- t-SNE Visualization ---------
    random_indices = np.random.permutation(data.shape[0])[0:400]
    for preprocess in [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(),
                       Normalizer(), PolynomialFeatures(), None]:
        new_data = preprocess.fit_transform()

        # 2D Visualization
        tsne = TSNE(n_components=2, verbose=1, perplexity=15, learning_rate=80, n_iter=500)
        results = tsne.fit_transform(new_data[random_indices,:])
        colors = np.array(['red' if label == 1 else 'greed' for label in labels[random_indices]])
        plt.scatter(results[:, 0], results[:, 1], c=colors)
        plt.title('t-SNE 2D Visualization' + str(preprocess).split('(')[0])
        plt.show()

    for preprocess in [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(),
                       Normalizer(), PolynomialFeatures(), None]:
        new_data = preprocess.fit_transform(data) if preprocess else data

        # 2D Visualization
        pca = PCA(n_components=2)
        results = pca.fit_transform(new_data)
        colors = np.array(['red' if label == 1 else 'green' for label in labels])
        results = results[random_indices, :]
        colors = colors[random_indices, :]
        plt.scatter(results[:, 0], results[:, 1], c=colors)
        plt.title('PCA Visualization 2D' + str(preprocess).split('(')[0])
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.show()
        plt.close()

        # 3D Visualization
        pca_3d = PCA(n_components=3)
        results_3d = pca_3d.fit_transform(new_data)
        colors_3d = np.array(['red' if label == 1 else 'green' for label in labels])
        plot_3d(results[random_indices,:], colors_3d[random_indices,:],'PCA 3D Visualization'+ str(preprocess).split('(')[0])

def split_data(train_portion, val_portion, data):
    val_begin = round(0.8 * M) ; val_end = round(val_begin + M * val_portion)

    train_set = data[: val_begin, :-1]
    val_set = data[val_begin : val_end, :-1]
    test_set = data[val_end :, :-1]

    train_labels = data[: val_begin, -1]
    val_labels = data[val_begin: val_end, -1]
    test_labels = data[val_end:, -1]

    data_dict = {'Training data' : train_set, 'Validation data': val_set, "Test data" : test_set, 'Training labels' : train_labels, 'Validation labels':val_labels, "Test labels": test_labels}
    return data_dict

def plot_data_distributions(train_set, train_labels, val_set, val_labels, test_set, test_labels):
    # Visualize Proportions of positive defaults in each set:
    grid = GridSpec(2, 2)
    train_defaults = sum(train_labels == 1) ; val_defaults = sum(val_labels == 1) ; test_defaults = sum(test_labels == 1)
    plt.subplot(grid[0, 0], aspect = 1, title= 'Training Set')
    plt.pie([train_defaults, train_set.shape[0] - train_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[1, 0], aspect = 1, title= 'Validation Set')
    plt.pie([val_defaults, val_set.shape[0] - val_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[0, 1], aspect = 1, title = "Test Set")
    plt.pie([test_defaults, test_set.shape[0] - test_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[1, 1], aspect = 1, title = "Aggregate Data")
    plt.pie([sum(data[:, -1] == 1), m - sum(data[:, -1])],  labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.show()

def build_nn(model_info):
    try:
        if model_info['Regularization'] == 'l2':
            lambda_ = model_info['Reg param']
            batch_norm, keep_prob = False, False

        elif model_info['Regularization'] == 'Batch norm':
            lambda_ = 0
            batch_norm = model_info['Reg param']
            keep_prob = False
            if batch_norm not in ['before', 'after']:
                raise ValueError

        elif model_info['Regularization'] == 'Dropout':
            lambda_, batch_norm = 0, False
            keep_prob = model_info['Reg param']
    except:
        lambda_, batch_norm , keep_prob = 0, False, False

    hidden, acts = model_info['Hidden layers'], model_info['Activations']
    model = Sequential(name=model_info['Name'])
    model.add(InputLayer((model_info['Input size'],)))
    first_hidden = True

    for lay, act , i in zip(hidden, acts, range(len(hidden))):
        if lambda_ > 0:
            if not first_hidden:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_), input_shape=(hidden[i - 1],)))
            else:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_),input_shape=(model_info['Input size'],)))
                first_hidden = False
        else:
            if not first_hidden:
                model.add(Dense(lay, input_shape=(hidden[i - 1],)))
            else:
                model.add(Dense(lay, input_shape=(model_info['Input size'],)))
                first_hidden = False
        if batch_norm == 'before':
            model.add(BatchNormalization(input_shape=(lay,)))
        model.add(Activation(act))

        if batch_norm == 'after':
            model.add(BatchNormalization(input_shape=(lay,)))

    # --------- Adding Output Layer -------------
    model.add(Dense(1,input_shape=(hidden[-1],)))
    if batch_norm == 'before':
        model.add(BatchNormalization(input_shape=(hidden[-1],)))
    model.add(Activation('sigmoid'))
    if batch_norm == 'after':
        model.add(BatchNormalization(input_shape=(hidden[-1],)))

    if model_info['Optimization'] == 'adagrad':
        opt = optimizers.Adagrad(lr=model_info['learning_rate'])
    elif model_info['Optimization'] == 'rmsprop':
        opt = optimizers.rmsprop(lr=model_info['learning_rate'])
    elif model_info['Optimization'] == 'adadelta':
        opt = optimizers.adadelta(lr=model_info['learning_rate'])
    elif model_info['Optimization'] == 'adamax':
        opt = optimizers.adamax(lr=model_info['learning_rate'])
    else:
        opt = optimizers.Nadam(lr=model_info['learning_rate'])

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def k_fold_cross_validation(param_dict, data, preproc_tech, model=None, num_epochs=15):
    kf = KFold(n_splits=5, shuffle=True)
    results = []

    for train_indices, test_indices in kf.split(data[:,:-1]):
        train_data , train_labels = preprocess_data(data[train_indices, :], preproc_tech, True)
        test_data, test_labels = preprocess_data(data[test_indices,:], preproc_tech,True)

        if not model:
            model = build_nn(param_dict)
        model.fit(train_data, train_labels, epochs=num_epochs, batch_size=param_dict['Batch size'],
                  verbose=0)
        y_pred = model.predict(train_data).ravel()
        fpr, tpr, thresholds = roc_curve(train_labels, y_pred)
        auc_train = auc(fpr, tpr)
        _, train_acc = model.evaluate(train_data, train_labels, verbose=0)

        y_pred = model.predict(test_data).ravel()
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
        auc_test = auc(fpr, tpr)
        _, test_acc = model.evaluate(test_data, test_labels, verbose=0)
        results.append(train_acc, test_acc, auc_train, auc_test)
    return results

def display_k_fold_results(model_hist, top_models = 1):
    """
    Displays results of k-fold cross validation test.
    :param model_hist:
    :param top_models: the number of top models to display (e.g. if you're testing 40 models, this will only show the top_models best performing models
    :return:
    """

    print('------------------ RESULTS FOR ' + model_hist[0]['Preprocessing'] + " PREPROCESSING ------------------------------------")
    top_test_auc = sorted(model_hist, key=lambda k: k['Avg Test AUC'])
    print('\nTOP Avg AUC Test:\n')
    for a_model in top_test_auc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_train_auc = sorted(model_hist, key=lambda k: k['Avg Train AUC'])
    print('\nTOP Avg AUC Train:\n\n')
    for a_model in top_train_auc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_test_acc = sorted(model_hist, key=lambda k: k["Avg Test Accuracy"])
    print('\nTOP Avg Test Accuracy:\n\n')
    for a_model in top_test_acc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_train_acc = sorted(model_hist, key=lambda k: k["Avg Train Accuracy"])
    print('\nTOP Avg Train Accuracy:\n\n')
    for a_model in top_train_acc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])
    print("------------------------------------------------------------------------------------------------------------")

def display_model_info(model_info):
    print('\n---------------------------------------------------------,')
    print("Architecture:\nLayers: ", [model_info['Input size']] + [model_info['Hidden layers']] + [1])                                                                   # show layers & number of units per layer
    print("Activations: ", model_info["Activations"] + ["sigmoid"])                                                                                                      # print hidden layer activations
    print("Hyperparameters:\nBatch size: ", model_info["Batch size"], "\nLearning rate: ", model_info["Learning rate"], "\nOptimization: ", model_info["Optimization"])
    if model_info["Preprocessing"]:
        print("Preprocessing: ", model_info["Preprocessing"])
    print("K-Fold CV RESULTS:\nAvg Train AUC: ", model_info["Avg Train AUC"], "\nAvg Test AUC: ", model_info["Avg Test AUC"], "\nAvg Train Accuracy: ", model_info["Avg Train Accuracy"])
    print("Avg Test Accuracy: ", model_info["Avg Test Accuracy"])

def generate_random_model():
    optimization_methods = ['adagrad', 'rmsprop', 'adadelta', 'adam', 'adamax', 'nadam']
    activation_functions = ['sigmoid', 'relu', 'tanh']
    batch_sizes = [16, 32, 64, 128, 512]
    range_hidden_units = range(5, 250)
    model_info = {}
    same_units = np.random.choice([0, 1], p=[1/5, 4/5])
    same_act_fun = np.random.choice([0, 1], p=[1/10, 9/10])

    really_deep = np.random.rand()
    range_layers = range(1, 10) if really_deep<0.8 else range(6, 20)

    num_layers = np.random.choice(range_layers, p=[.1, .2, .2, .2, .05, .05, .05, .1, .05]) if really_deep < 0.8 \
                 else np.random.choice(range_layers)
    model_info["Activations"] = [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25])] * num_layers \
                              if same_act_fun else [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25])
                              for _ in range(num_layers)] # choose activation functions
    model_info["Hidden layers"] = [np.random.choice(range_hidden_units)] * num_layers \
                                 if same_units else [np.random.choice(range_hidden_units) for _ in range(num_layers)]  # create hidden layers
    model_info["Optimization"] = np.random.choice(optimization_methods)         # choose an optimization method at random
    model_info["Batch size"] = np.random.choice(batch_sizes)                    # choose batch size
    model_info["Learning rate"] = 10 ** (-4 * np.random.rand())                 # choose a learning rate on a logarithmic scale
    model_info["Training threshold"] = 0.5                                      # set threshold for training
    return model_info

def quick_nn_test(model_info, data_dict, save_path):
    model = build_nn(model_info)
    stop = EarlyStopping(patience=5, monitor='acc', verbose=1)
    tensorboard_path = save_path + model_info['Name']
    tensorboard = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True)
    save_model = ModelCheckpoint(filepath=save_path+model_info['Name']+'/'+model_info['Name']+'_saved'+'.h5')

    model.fit(data_dict['Training_data'], data_dict['Training_labels'], epochs=3,
              batch_size=model_info['Batch_size'], callbacks=[save_model,stop,tensorboard])
    train_acc = model.evaluate(data_dict['Training_data'],data_dict['Training_labels'],
                               batch_size=model_info['Batch_size'],verbose=0)
    test_acc = model.evaluate(data_dict['Test_data'],data_dict['Test_labels'],
                              batch_size=model_info['Batch_size'], verbose=0)
    y_pred = model.predict(data_dict['Training_data']).ravel()
    fpr,tpr,thresholds = roc_curve(data_dict['Training_labels'],y_pred)
    auc_train = auc(fpr,tpr)

    y_pred = model.predict(data_dict['Test_data']).ravel()
    fpr,tpr,thresholds = roc_curve(data_dict['Test_data'], y_pred)
    auc_test = auc(fpr, tpr)

    return train_acc, test_acc, auc_train, auc_test

def test_nn_models(num_models, raw_data, preprocess_tech, save_path, eval_tech='k-fold',
                   list_of_model=None):
    model_hist = []
    model_results = pd.DataFrame(
        columns=['Hidden', 'Activations', 'Learning rate', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC'])
    result_index = 0
    if eval_tech not in ['k-fold', 'boot']:
        raise NameError
    for i in range(num_models):
        if not list_of_model:
            model_info = generate_random_model()
            model_info['Preprocessing'] = preprocess_tech
            model_info['Input size'] = raw_data.shape[1] - 1
        else:
            assert num_models == len(list_of_model)
            model_info = list_of_model[i]

        model_info['Model type'] = 'NN'
        model = build_nn(model_info)

        if eval_tech == 'k-fold':
            results = k_fold_cross_validation(model_info, raw_data, preprocess_tech, model=model)
        else:
            print("currently not supporting bootstrap evaluation due to computational constraints")

        train_acc, test_acc, train_auc, test_auc = zip(*results)
        model_info["Avg Test AUC"] = round(np.mean(test_auc), 3)                                            # compute average metrics for each fold of the evaluation
        model_info["Avg Train AUC"] = round(np.mean(train_auc), 3)
        model_info["Avg Train Accuracy"] = round(np.mean(train_acc), 3)                                     # these are averages from k-fold cross validation
        model_info["Avg Test Accuracy"] = round(np.mean(test_acc), 3)
        model_hist.append(model_info)
        model_results.loc[result_index] = (model_info['Hidden layers'], model_info['Activations'],\
                                           model_info['Learning rate'], model_info['Avg Train Accuracy'],\
                                           model_info["Avg Test Accuracy"], \
                                           model_info["Avg Train AUC"], model_info["Avg Test AUC"])
        result_index += 1
    model_results.to_csv(save_path+'K-fold NN Results.csv')
    display_k_fold_results(model_hist,top_models=1)

def create_five_nns(input_size, hidden_size, act=None):
    """
    Creates 5 neural networks to be used as a baseline in determining the influence model depth & width has on performance.
    :param input_size:
    :param hidden_size:
    :param act: activation function to use for each layer
    :return:
    """
    act = ['relu'] if not act else [act]                             # default activation = 'relu'
    nns = []                                                         # list of model info hash tables
    model_info = {}                                                  # hash tables storing model information
    model_info['Hidden layers'] = [hidden_size]
    model_info['Input size'] = input_size
    model_info['Activations'] = act
    model_info['Optimization'] = 'adadelta'
    model_info["Learning rate"] = .005
    model_info["Batch size"] = 32
    model_info["Preprocessing"] = 'Standard'
    model_info2, model_info3, model_info4, model_info5 = model_info.copy(), model_info.copy(), model_info.copy(), model_info.copy()

    model_info["Name"] = 'Shallow NN'                                 # build shallow nn
    nns.append(model_info)

    model_info2['Hidden layers'] = [hidden_size] * 3                  # build medium nn
    model_info2['Activations'] = act * 3
    model_info2["Name"] = 'Medium NN'
    nns.append(model_info2)

    model_info3['Hidden layers'] = [hidden_size] * 6                  # build deep nn
    model_info3['Activations'] = act * 6
    model_info3["Name"] = 'Deep NN 1'
    nns.append(model_info3)

    model_info4['Hidden layers'] = [hidden_size] * 11                 # build really deep nn
    model_info4['Activations'] = act * 11
    model_info4["Name"] = 'Deep NN 2'
    nns.append(model_info4)

    model_info5['Hidden layers'] = [hidden_size] * 20                   # build realllllly deep nn
    model_info5['Activations'] = act * 20
    model_info5["Name"] = 'Deep NN 3'
    nns.append(model_info5)
    return nns

# ------------------- ESTABLISH BASELINES w/ 5-FOLD CROSS VALIDATION -------------------
# Test Logistic regression on original data preprocessed and og_one_hot preprocessed
og_one_hot = np.array(pd.read_csv(''))

# print("TESTING.....")

# Evaluate several models via 5-Fold Cross validation
save_path = r'YOUR PATH HERE'

# ------------------------ QUICK TESTING ------------------------

"""This section of code allows us to create and test many neural networks and save the results of a quick 
test into a CSV file. Once that CSV file has been created, we will continue to add results onto the existing 
file."""

rapid_testing_path = 'YOUR PATH HERE'  # TODO: UNCOMMENT THIS
data_path = 'YOUR DATA PATH' # TODO: UNCOMMENT THIS


try:                                                                        # try to load existing csv
    rapid_mlp_results = pd.read_csv(rapid_testing_path + 'Results.csv')
    index = rapid_mlp_results.shape[1]
except:                                                                     # if no csv exists yet, create a DF
    rapid_mlp_results = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC',
                                              'Preprocessing', 'Batch size', 'Learn Rate', 'Optimization', 'Activations',
                                              'Hidden layers', 'Regularization'])
    index = 0

og_one_hot = np.array(pd.read_csv(data_path))                     # load one hot data

model_info = {}                                                     # create model_info dicts for all the models we want to test
model_info['Hidden layers'] = [100] * 6                             # specifies the number of hidden units per layer
model_info['Input size'] = og_one_hot.shape[1] - 1                  # input data size
model_info['Activations'] = ['relu'] * 6                            # activation function for each layer
model_info['Optimization'] = 'adadelta'                             # optimization method
model_info["Learning rate"] = .005                                  # learning rate for optimization method
model_info["Batch size"] = 32
model_info["Preprocessing"] = 'Standard'                            # specifies the preprocessing method to be used

model_0 = model_info.copy()                                         # create model 0
model_0['Name'] = 'Model0'

model_1 = model_info.copy()                                         # create model 1
model_1['Hidden layers'] = [110] * 3
model_1['Name'] = 'Model1'

model_2 = model_info.copy()                                         # try best model so far with several regularization parameter values
model_2['Hidden layers'] = [110] * 6
model_2['Name'] = 'Model2'
model_2['Regularization'] = 'l2'
model_2['Reg param'] = 0.0005

model_3 = model_info.copy()
model_3['Hidden layers'] = [110] * 6
model_3['Name'] = 'Model3'
model_3['Regularization'] = 'l2'
model_3['Reg param'] = 0.05

model_4 = model_info.copy()                                                             # try best model so far with several regularization parameter values
model_4['Hidden layers'] = [110] * 6
model_4['Name'] = 'Model4'
model_4['Regularization'] = 'l2'
model_4['Reg param'] = 0.0005

# .... create more models ....

#-------------- REGULARIZATION OPTIONS -------------
#   L2 Regularization:      Regularization: 'l2',           Reg param: lambda value
#   Dropout:                Regularization: 'Dropout',      Reg param: keep_prob
#   Batch normalization:    Regularization: 'Batch norm',   Reg param: 'before' or 'after'

models = [model_0, model_1, model_2]                                  # make a list of model_info hash tables

column_list = ['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC', 'Preprocessing',
               'Batch size', 'Learn Rate', 'Optimization', 'Activations', 'Hidden layers',
               'Regularization', 'Reg Param']

for model in models:                                                                                          # for each model_info in list of models to test, test model and record results
    train_data, labels = preprocess_data(og_one_hot, model['Preprocessing'], True)                            # preprocess raw data
    data_dict = split_data(0.9, 0, np.concatenate((train_data, labels.reshape(29999, 1)), axis=1))             # split data
    train_acc, test_acc, auc_train, auc_test = quick_nn_test(model, data_dict, save_path=rapid_testing_path)  # quickly assess model

    try:
        reg = model['Regularization']                                             # set regularization parameters if given
        reg_param = model['Reg param']
    except:
        reg = "None"                                                              # else set NULL params
        reg_param = 'NA'

    val_lis = [model['Name'], train_acc[1], test_acc[1], auc_train, auc_test, model['Preprocessing'],
                model["Batch size"], model["Learning rate"], model["Optimization"], str(model["Activations"]),
                str(model["Hidden layers"]), reg, reg_param]
    df_dict = {}
    for col, val in zip(column_list, val_lis):                                    # create df dict to append to csv file
        df_dict[col] = val

    df = pd.DataFrame(df_dict, index=[index])
    rapid_mlp_results = rapid_mlp_results.append(df, ignore_index=False)
    rapid_mlp_results.to_csv(rapid_testing_path + "Results.csv", index=False)

