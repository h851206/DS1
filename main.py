# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import TingYu.Methods
import TingYu.measurement_table

def wine_quality():
    red_wine_dir = '/Users/ting-yuho/Desktop/DS1/finalProject/redwine-quality.csv'
    white_wine_dir = '/Users/ting-yuho/Desktop/DS1/finalProject/redwine-quality.csv'
    raw_df, count0 = TingYu.Methods.combine_raw_data(data_red_dir=red_wine_dir, data_white_dir=white_wine_dir)
    st.header('Wine Quality data set')
    st.dataframe(raw_df)

    option = st.sidebar.selectbox(
        'Method selection (wine quality data)',
        ['without undersampling', 'Random selection and Kmean++', 'Random sampling', 'n near neighbours and Kmean++',
         'centroid and Kmean++', ''])
    st.header('selected method：' + option)
    if option == 'without undersampling':
        acc, df_plot, train_df, AUC = TingYu.Methods.imbalance(raw_df)
        st.dataframe(train_df)
        st.text("Multilabel AUC")
        st.dataframe(AUC)
    elif option == 'Random selection and Kmean++':
        num_cluster = st.number_input("Number of cluster")
        if num_cluster>1:
            num_cluster = np.floor(num_cluster)
            num_cluster = int(num_cluster)
            aacc, df_plot, train_df, AUC = TingYu.Methods.RS_Kmean(raw_df, n_clusters=num_cluster)
            expander = st.beta_expander("Click to check feature table...")
            expander.write(train_df)
            expander.write("Multilabel AUC")
            expander.write(AUC)

            # plot
            feature_option = st.selectbox('Select', ['precision', 'recall', 'f1-score', 'AUC'])
            f, ax = plt.subplots(figsize=(14, 8))
            if feature_option == 'AUC':
                ax = TingYu.measurement_table.bar(AUC, 'index', feature_option, ax, f, feature_option, 'Wine Quality',
                                                  option)
            else:
                ax = TingYu.measurement_table.bar(df_plot, 'index', feature_option, ax, f, feature_option, 'Wine Quality',
                                                  option)
            st.pyplot(f)
        else:
            st.text('Please given a cluster number larger than one!')
    elif option == 'Random sampling':
        acc, df_plot, train_df, AUC = TingYu.Methods.RS(raw_df)
        st.dataframe(train_df)
        st.text("Multilabel AUC")
        st.dataframe(AUC)

if __name__ == '__main__':
    st.title('DS1 T2-1')
    st.text("Group members: TingYu")
    df_option = st.sidebar.selectbox(
        'Data set selection',
        ['Logistic Regression and UCI Bank Marketing dataset', 'Gradient Boosting Decision Tree and Adult Data Set',
         'Balance Scale Data Set with Naive Bayes', 'Wine Quality with SVM'])
    if df_option == 'Logistic Regression and UCI Bank Marketing dataset':
        # WenXu's place
        pass

    elif df_option == 'Gradient Boosting Decision Tree and Adult Data Set':
        # Phillip's place
        pass

    elif df_option == 'Balance Scale Data Set with Naive Bayes':
        # Max's place
        pass

    elif df_option == 'Wine Quality with SVM':
        wine_quality()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
