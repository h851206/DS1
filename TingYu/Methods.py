import measurement_table
import pandas as pd


def imbalance(data):
    X_train, y_train, X_test, y_test = measurement_table.train_test(data)
    acc1, df1, train_df1, AUC1 = measurement_table.train_classifier(X_train, y_train, X_test, y_test, 550)
    df1_plot = measurement_table.transform(df1)
    return acc1, df1_plot, train_df1, AUC1

def RS_Kmean(data):
    X_train, y_train, X_test, y_test = measurement_table.train_test(data)
    X_train = measurement_table.clustering(X_train, 6)
    X_train['index'] = X_train.index
    kmean_y_train = X_train['kmeans_label']
    X_train_miss, y_train_miss = measurement_table.undersampling(X_train, kmean_y_train)
    y_train_miss = y_train.loc[X_train_miss['index']]
    X_train_miss = X_train_miss.drop(columns=['kmeans_label','index'])
    y_train_miss.value_counts()
    count1 = y_train_miss.to_frame()
    count1['method'] = 'Random selection after Kmean++'
    acc2, df2, train_df2, AUC2 = measurement_table.train_classifier(X_train_miss, y_train_miss, X_test, y_test, 550)
    df2_plot = measurement_table.transform(df2)
    return acc2, df2_plot, train_df2, AUC2

def RS(data):
    X_train, y_train, X_test, y_test = measurement_table.train_test(data)
    X_train_miss, y_train_miss = measurement_table.undersampling(X_train, y_train)
    count2 = pd.DataFrame()
    count2['quality'] = y_train_miss
    count2['method'] = 'Random selection'
    acc3, df3, train_df3, AUC3 = measurement_table.train_classifier(X_train_miss, y_train_miss, X_test, y_test, 550)
    df3_plot = measurement_table.transform(df3)
    return acc3, df3_plot, train_df3, AUC3

def n_near_Kmean(data, n):
    X_train, y_train, X_test, y_test = measurement_table.train_test(data)
    X_train_neighbor, y_train_neighbor = measurement_table.N_neighbor(X_train, y_train, 6, n)
    count3 = pd.DataFrame()
    count3['quality'] = y_train
    count3['method'] = 'Top N neighbors and Kmean++'
    acc4, df4, train_df4, AUC4 = measurement_table.train_classifier(X_train_neighbor, y_train_neighbor, X_test, y_test, 550)
    df4_plot = measurement_table.transform(df4)
    return acc4, df4_plot, train_df4, AUC4

def centroid_Kmean(data):
    X_train, y_train, X_test, y_test = measurement_table.train_test(data)
    X_train, y_train, y_test = measurement_table.centroid(X_train, X_test, 7)
    count5 = pd.DataFrame()
    count5['quality'] = y_train+3
    count5['method'] = 'centroid'

    acc6, df6, train_df6, AUC6 = measurement_table.train_classifier(X_train, y_train, X_test, y_test, 550)
    df6_plot = measurement_table.transform(df6)
    df6_plot['index'] = df6_plot['index'].astype('int')
    df6_plot['index'] = df6_plot['index']+3
    return acc6, df6_plot, train_df6, AUC6