import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import altair as alt
import streamlit as st
from sklearn import preprocessing
import cleanup as cp
import umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def load_data():
    # cp.prep_fbi_dataset()
    df = cp.prep_fbi_dataset()
    return df

def plot_cluster(selection):
    #partition dataframe depending on 
    if selection == 'Bias':
        X = df.iloc[:,75:110].to_numpy()
        labels = df.columns[75:110]
    elif selection == 'Crime':
        X = df.iloc[:,18:75].to_numpy()
        labels = df.columns[18:75]
    elif selection == 'Location':
        X = df.iloc[:,110:].to_numpy()
        labels = df.columns[110:]

    np.random.seed(0)
    random_sample = np.random.permutation(len(X))[:5000]
    X = X[random_sample]

    #Dimensionality Reduction
    X_tsne2d = umap.UMAP(densmap=True,n_components=2,random_state=0).fit_transform(X)

    #Clustering
    dbscan = DBSCAN(min_samples=200, eps=1)
    dbscan_cluster_assignments = dbscan.fit_predict(X_tsne2d)
    inliers = dbscan_cluster_assignments != -1

    #plotting the cluster
    data = pd.concat([pd.DataFrame(X_tsne2d[:,:][inliers],columns=['X','Y']), pd.DataFrame(dbscan_cluster_assignments[inliers],columns=['Assign'])], axis=1)
    total_counts = data['Assign'].value_counts()
    assign = data['Assign']
    data['total_count'] = [total_counts.get(a) for a in assign]
    data['common_indicator'] = [labels[np.argmax(X[dbscan_cluster_assignments == i].sum(axis=0))] for i in dbscan_cluster_assignments[inliers]]
    chart = alt.Chart(data,title='Clustering Bias').mark_point().encode(
        alt.X("X"),
        alt.Y("Y"),
        alt.Color("Assign:N",legend=alt.Legend(title="Clusters")),
        alt.Tooltip(["Assign",'total_count','common_indicator'])
    )

    return chart
    
st.title("Application")
with st.spinner(text="Loading data..."):
    df = load_data()
st.write(df.head())

selection = st.selectbox("Select feature",options = ['Bias','Crime','Location'])

if selection:
    cluster = plot_cluster(selection)
    
st.write(selection)
st.write(cluster)





