import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import NearMiss
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import ptitprince as pt
from sklearn import metrics

def combine_raw_data(
        data_red_dir: pd.DataFrame, data_white_dir: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    data_white = pd.read_csv(data_white_dir)
    data_red = pd.read_csv(data_red_dir)

    data_white['wine_class'] = -1
    data_red['wine_class'] = 1

    data = pd.concat([data_red, data_white], ignore_index=True)
    count0 = pd.DataFrame()
    count0['quality'] = data['quality']
    count0['method'] = 'imbalance dataset'
    return data, count0


def accuracy(y_test, y_predict):
    count = 0
    for test, predict in zip(y_test, y_predict):
        if test == predict:
            count = count + 1
    acc = count / len(y_test)
    return acc


def train_test(data):
    # make results reproducible
    np.random.seed(42)
    # sample without replacement
    train_ix = np.random.choice(data.index, int(len(data) * 0.8), replace=False)
    df_training = data.loc[train_ix]
    df_test = data.drop(train_ix)

    X_train = df_training.drop(columns=['quality'])
    Y_train = df_training['quality']
    X_test = df_test.drop(columns=['quality'])
    Y_test = df_test['quality']
    return X_train, Y_train, X_test, Y_test


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def train_classifier(X_train, Y_train, X_test, Y_test, max_iter=300):
    #     clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=max_iter)
    #     clf.fit(X_train, Y_train)
    #     Y_predict = clf.predict(X_test)
    model = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=max_iter)).fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    acc = accuracy(Y_test, Y_predict)
    report = pd.DataFrame(classification_report(Y_test, Y_predict, output_dict=True))

    # training evaluation
    preds = model.predict(X_train)
    targs = Y_train
    AUC = pd.DataFrame.from_dict(roc_auc_score_multiclass(targs, preds), orient='index')
    AUC['index'] = AUC.index
    AUC = AUC.rename(columns={0: 'AUC'})
    train_df = pd.DataFrame({"accuracy": [metrics.accuracy_score(targs, preds)],
                             "precision": [metrics.precision_score(targs, preds, average='micro')],
                             "recall": [metrics.recall_score(targs, preds, average='micro')],
                             "f1": [metrics.f1_score(targs, preds, average='micro')]
                             })
    print(train_df)
    print(AUC)
    return acc, report, train_df, AUC


def undersampling(X_train, y_train):
    nr = NearMiss()
    X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel())
    return X_train_miss, y_train_miss


def centroid(X_train, X_test, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(X_train)
    y_train = np.arange(0, n_clusters)
    centroid = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(X_test)
    y_test = kmeans.labels_
    return centroid, y_train, y_test


def clustering(X_train, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(X_train)
    X_train['kmeans_label'] = kmeans.labels_
    return X_train


def bar(df, x_axis, y_axis, ax, fig, xlabel, ylabel, title):
    sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax, palette="Set3")
    fig.canvas.draw()
    ax.set_ylabel(xlabel, fontsize=18)
    ax.set_xlabel(ylabel, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.set_title(title, fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


def transform(df):
    select = df.keys().tolist()
    select = select[:-3]
    df = df[select].transpose()
    return df.reset_index()


def make_raincloud_plot(
        data: pd.DataFrame, x_axis: str, y_axis: str, ort: str = 'v'
):
    """raincloud plotting function

    Parameters
    ----------
    data: pandas Dataframe
        The data needed to be plotted
    x_axis: str
        the name of variable for x axis
    y_axis: str
        the name of variable for y axis
    ort: str
        decide the orientaiton of graph 'v' or 'h'

    Returns
    -------
    f: Figure
        the plot
    ax: axes.Axes or array of Axes
        ax can be either a single Axes object or an array of Axes objects if more than one subplot was created.

    """
    f, ax = plt.subplots(figsize=(9, 6))
    ax = pt.RainCloud(
        x=x_axis,
        y=y_axis,
        data=data,
        palette="Set2",
        bw=0.2,
        width_viol=0.5,
        ax=ax,
        orient=ort,
    )
    f.tight_layout()
    return f, ax


def data_distrbution_concat(datalist):
    distr_plot_df = pd.concat(datalist, ignore_index=True)
    f, ax = make_raincloud_plot(distr_plot_df, 'method', 'quality', 'h')
    ax.set_ylabel('Method', fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.set_xlabel('Wine Quality', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_title('data distribution', fontsize=18)

    f.show()


def N_neighbor(X_train, y_train, n_clusters, n_neighbor):
    X_train = X_train.reset_index()
    sample = X_train.drop(columns=['index'])

    centroid_df = pd.DataFrame()
    cluster_label = np.arange(0, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(sample)
    neigh = NearestNeighbors(n_neighbors=n_neighbor)
    neigh.fit(sample)

    centroid = kmeans.cluster_centers_

    # n_clusters* n_feature
    index = neigh.kneighbors(centroid, n_neighbor, return_distance=False)
    index = np.unique(index.ravel())

    select_sample = sample.loc[index]

    new_X_train = select_sample.merge(X_train)
    new_y_train = y_train.loc[new_X_train['index'].tolist()]

    return new_X_train.drop(columns=['index']), new_y_train


def make_seaborn_scatter_plot(
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        plot_type: str,
        hue: str,
        fontsize: int,
        xfrequency: int = None,
        figsize: tuple = (10, 5),
):
    """

    Parameters
    ----------
    data: pandas Dataframe
        The data needed to be plotted
    x_axis: str
        the name of variable for x axis
    y_axis: str
        the name of variable for y axis
    plot_type: str
        the plot type such as error_bar, scatter, strip
    hue: str
        name of column which contain info about separating data
    fontsize: int
        fontsize of xlabel
    xfrequency: int
        frequncy of sampling xtickslabel. fequency =2: [0,1,2,3,4,5,6,7,8] -> [0,2,4,6,8]
    figsize: tuple
        shape of plot

    Returns
    -------
    f: Figure
        the plot
    axs: axes.Axes or array of Axes
        ax can be either a single Axes object or an array of Axes objects if more than one subplot was created.

    """
    f, ax = plt.subplots(figsize=figsize)
    base_kwargs = {'x': x_axis, 'y': y_axis, 'ax': ax}
    if plot_type == 'scatter':
        base_kwargs.update({'hue': hue, 'alpha': 1, 'data': data})
        sns.scatterplot(**base_kwargs)
    elif plot_type == 'strip':
        base_kwargs.update({'hue': hue, 'data': data, 'size': 12})
        sns.stripplot(**base_kwargs)
    elif plot_type == 'error_bar':
        sns.set_theme(style="darkgrid")
        base_kwargs.update(
            {
                'hue': hue,
                'data': data,
                'ci': 'sd',
                'join': False,
                'dodge': 0.7,
                'errwidth': 0.6,
            }
        )
        sns.pointplot(**base_kwargs)
    ax.set_xlabel(
        r'max follicle volume ($mm^3$) for each day each patient',
        fontsize=fontsize,
    )
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 1.7)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(loc='lower right', fontsize=10)
    f.canvas.draw()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    if xfrequency:
        for label in ax.get_xticklabels()[::xfrequency]:
            label.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    f.tight_layout()
    return f, ax

