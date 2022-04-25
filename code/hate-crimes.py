import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import cleanup as cp
import umap
from sklearn.cluster import DBSCAN


@st.cache()
def load_data():
    df = cp.prep_fbi_dataset()
    df_city = cp.prep_city_dataset()
    features_final=pd.read_csv("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")
    return df,df_city,features_final

@st.cache(allow_output_mutation=True)
def plot_cluster(selection):
    #partition dataframe depending on selection
    bias = df.iloc[:,75:110]
    bias_labels = df.columns[75:110]

    crime = df.iloc[:,18:66]
    crime_labels = df.columns[18:66]

    victim_type = df.iloc[:,66:75]
    victim_type_labels = df.columns[66:75]

    location = df.iloc[:,110:]
    location_labels = df.columns[110:]

    offender = df.iloc[:,10:18]
    offender_labels = df.columns[10:18]
    offender_labels = np.array([i[14:] for i in offender_labels])


    num_indicators = len(selection)
    if num_indicators == 1:
        if selection[0] == 'Bias':
            X = bias.to_numpy()
            labels = bias_labels
        elif selection[0] == 'Crime':
            X = crime.to_numpy()
            labels = crime_labels
        elif selection[0] == 'Location':
            X = location.to_numpy()
            labels = location_labels
        elif selection[0] == 'Offender Race':
            X = offender.to_numpy()
            labels = offender_labels
        elif selection[0] == 'Victim Type':
            X = victim_type.to_numpy()
            labels = victim_type_labels
    else:
        for idx,s in enumerate(selection):
            if idx == 0:
                if s == 'Bias':
                    X = bias.to_numpy()
                    labels = bias_labels
                elif s == 'Crime':
                    X = crime.to_numpy()
                    labels = crime_labels
                elif s == 'Location':
                    X = location.to_numpy()
                    labels = location_labels
                elif s == 'Offender Race':
                    X = offender.to_numpy()
                    labels = offender_labels
                elif s == 'Victim Type':
                    X = victim_type.to_numpy()
                    labels = victim_type_labels
            else:
                if s == 'Bias':
                    X = np.concatenate((X,bias.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,bias_labels))
                elif s == 'Crime':
                    X = np.concatenate((X,crime.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,crime_labels))
                elif s == 'Location':
                    X = np.concatenate((X,location.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,location_labels))
                elif s == 'Offender Race':
                    X = np.concatenate((X,offender.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,offender_labels))
                elif s == 'Victim Type':
                    X = np.concatenate((X,victim_type.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,victim_type_labels))
    

    np.random.seed(0)
    random_sample = np.random.permutation(len(X))[:1000]
    X = X[random_sample]

    #Dimensionality Reduction
    X_tsne2d = umap.UMAP(densmap=True,n_components=2,n_neighbors=2,random_state=0).fit_transform(X)

    #Clustering
    dbscan = DBSCAN(min_samples=10, eps=1)
    dbscan_cluster_assignments = dbscan.fit_predict(X_tsne2d)
    inliers = dbscan_cluster_assignments != -1

    #plotting the cluster
    data = pd.concat([pd.DataFrame(X_tsne2d[:,:][inliers],columns=['X','Y']), pd.DataFrame(dbscan_cluster_assignments[inliers],columns=['Assign'])], axis=1)
    total_counts = data['Assign'].value_counts()
    assign = data['Assign']
    data['total_count'] = [total_counts.get(a) for a in assign]
    if len(selection) == 1:
        data['common_indicator'] = [labels[np.argmax(X[dbscan_cluster_assignments == i].sum(axis=0))] for i in dbscan_cluster_assignments[inliers]]
    else:
        data['common_indicator'] = [labels[np.sort(np.argsort(-X[dbscan_cluster_assignments == i].sum(axis=0))[0:(num_indicators)])].tolist() for i in dbscan_cluster_assignments[inliers]]
    chart = alt.Chart(data,title='{}'.format(selection)).mark_point().encode(
        alt.X("X"),
        alt.Y("Y"),
        alt.Color("Assign:N",legend=alt.Legend(title="Clusters")),
        alt.Tooltip(["Assign",'total_count','common_indicator'])
    ).properties(
    width=500,
    height=500)

    return chart
    
st.title("Hate Crimes in the United States")

with st.spinner(text="Loading data..."):
    df,df_city,features_final = load_data()

st.write("FBI Hate Crimes Dataset")    
st.write(df.head())

st.write("Cities Dataset")
st.write(df_city.head())

additional = st.checkbox('Would you like to view additional data?')

if additional:
    st.selectbox("Select your features",options=["Wellness Factor"])
    
 #Clustering   
selection = st.multiselect("Select your features",options = ['Bias','Crime','Location','Offender Race','Victim Type'])

#make selection for clustering
if selection:
    cluster = plot_cluster(selection)
    st.write(cluster)



#Feature Importance

# def load_data(name):
#     return pd.read_csv(name)


# data = load_data("features_final.csv")

state = st.text_input("Enter 2 letter state abbreviation")

#City Visualization
df = features_final
#selecting only the cities in the selected state
df.drop(df[df['STATE_ABBR'] != state].index, inplace = True) 
st.write("Enter one of the following cities in " + state)
st.write(df['City'])

#select the city for which you want to see the factors
city = st.text_input("Enter city name")
if( city in df.values  ):
    df.drop(df[df['City'] != city].index, inplace = True)
    print(df.head())
    source = pd.DataFrame({
        'X' : ['High School Completion','Income Inequality','Life Expectancy','Racial Segregation','Racial Diversity','Unemployment','Uninsured'],
        'Y': [df['High School Completion'].tolist()[0],df['Income Inequality'].tolist()[0], df['Life Expectancy'].tolist()[0], df['Neighborhood racial/ethnic segregation'].tolist()[0], df['Racial/ethnic diversity'].tolist()[0], df['Unemployment - annual, neighborhood-level'].tolist()[0], df['Uninsured'].tolist()[0] ],
        })
    hist = alt.Chart(source).mark_bar().encode(
        x='X',
        y='Y'
    ).properties(
        width=500,
        height=800)
        
    st.header("Visualizing features in a city")
    st.write(hist)
else:
    st.write("Please choose a city from th list provided")
    
