import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import cleanup as cp
import umap
from vega_datasets import data
from sklearn.cluster import DBSCAN


@st.cache(allow_output_mutation=True)
def load_data():
    df = cp.prep_fbi_dataset()
    df_city = cp.prep_city_dataset()
    return df,df_city

@st.cache(allow_output_mutation=True)
def plot_cluster(selection):
    #partition dataframe depending on selection
    bias = df_hate.iloc[:,81:116]
    bias_labels = df_hate.columns[81:116]

    crime = df_hate.iloc[:,24:72]
    crime_labels = df_hate.columns[24:72]

    victim_type = df_hate.iloc[:,72:81]
    victim_type_labels = df_hate.columns[72:81]

    location = df_hate.iloc[:,116:]
    location_labels = df_hate.columns[116:]

    offender = df_hate.iloc[:,16:24]
    offender_labels = df_hate.columns[16:24]
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
    df_hate,df_city = load_data()

st.write("FBI Hate Crimes Dataset")    
st.write(df_hate.head())

st.write("Cities Dataset")
st.write(df_city.head())

additional = st.checkbox('Would you like to view additional data?')

if additional:
    option = st.selectbox("Select your features",options=['Hate Crimes',"Wellbeing of Cities"])

    if option == 'Hate Crimes':
        cities = alt.Chart(df_hate.head(1000)).mark_bar().encode(
        x=alt.X("STATE_NAME",  sort="-y"),
        y=alt.Y("count()")
        )

        st.write(cities)

        # Offender Race 
        offenderRace = alt.Chart(df_hate.head(1000)).mark_bar().encode(
            x=alt.X("OFFENDER_RACE",  sort="-y"),
            y=alt.Y("count()")
        )

        # Victim Race 
        victimRace = alt.Chart(df_hate.head(1000)).mark_bar().encode(
            x=alt.X("BIAS_DESC",  sort="-y"),
            y=alt.Y("count()")
        ).transform_window(
            rank='rank(BIAS_DESC)'
        ).transform_filter(
            (alt.datum.rank < 1000)
        )

        st.write(offenderRace|victimRace) #| st.write(victimRace)

        #Line graph of total recorded crimes per year
        chart = alt.Chart(df_hate.head(5000)).mark_line().encode(
            alt.X('DATA_YEAR'),
            alt.Y('VICTIM_COUNT', aggregate='sum')
        )
        chart.encoding.x.title='Crime Year'
        chart.encoding.y.title='Total Recorded Crime Count'
        st.write(chart)
    else:
        #Plotting High School Completion 
        # df_city_expl = pd.read_csv("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/city_data.csv")
        # st.write(df_city)
        data_final1  = df_city[df_city["metric_name"].isin(["High school completion"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        High_School_Completion_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[85,100]))
        ).properties(
        title = "High School Completion Well-Being Factor across States")

        st.write(High_School_Completion_bar)

        #Plotting Life expectancy 
        data_final1  = df_city[df_city["metric_name"].isin(["Life expectancy"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Life_expectancy_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[74,86]))
        ).properties(
        title = "Life expectancy Well-Being Factor across States")
        
        st.write(Life_expectancy_bar)


        #Plotting Income Inequality 
        data_final1  = df_city[df_city["metric_name"].isin(["Income Inequality"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Income_Inequality_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,60]))
        ).properties(
        title = "Income Inequality Well-Being Factor across States")

        st.write(Income_Inequality_bar)


        #Plotting Neighborhood racial/ethnic segregation
        data_final1  = df_city[df_city["metric_name"].isin(["Neighborhood racial/ethnic segregation"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Neighborhood_racial_ethnic_segregation_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,45]))
        ).properties(
        title = "Neighborhood racial/ethnic segregation Well-Being Factor across States")

        st.write(Neighborhood_racial_ethnic_segregation_bar)

        #Plotting Racial/ethnic diversity
        data_final1  = df_city[df_city["metric_name"].isin(["Racial/ethnic diversity"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Racial_ethnic_diversity_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[35,100]))
        ).properties(
        title = "Racial/ethnic diversity Well-Being Factor across States")

        st.write(Racial_ethnic_diversity_bar)

    
        #Plotting Unemployment Factor
        data_final1  = df_city[df_city["metric_name"].isin(["Unemployment - annual, neighborhood-level"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Unemployment_Factor_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,20]))
        ).properties(
        title = "Unemployment Factor Well-Being Factor across States")

        st.write(Unemployment_Factor_bar)


#US MAP
# Title 
st.header("CRIME SCENE")

# Importing Data
df =pd.read_csv("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")
alt.data_transformers.disable_max_rows()
df_HeatMap = df[['BIAS_DESC','OFFENDER_RACE']].copy()

#Drawing up the US MAP
st.subheader("US MAP")

#Adding a new column Frequency, that holds the number of cases in each state.
df['Frequency']=df['STATE_NAME'].map(df['STATE_NAME'].value_counts())
df = df.drop_duplicates(subset=['STATE_NAME'], keep='first')
df = df.rename({'STATE_NAME': 'state'}, axis = 1)

#Alinging the state values
ansi = pd.read_csv('https://www2.census.gov/geo/docs/reference/state.txt', sep='|')
ansi.columns = ['id', 'abbr', 'state', 'statens']
ansi = ansi[['id', 'abbr', 'state']]
df = pd.merge(df, ansi, how='left', on='state')
states = alt.topo_feature(data.us_10m.url, 'states')


#Defining selection criteria 
click = alt.selection_multi(fields=['state'])

# Building and displaying the US MAP
displayUSMap = alt.Chart(states).mark_geoshape().encode(
    color=alt.Color('Frequency:Q', title = "No. of cases"),         
    tooltip=['Frequency:Q', alt.Tooltip('Frequency:Q')],    
    opacity = alt.condition(click, alt.value(1), alt.value(0.2)),
).properties(
    title = "Choropleth Map of the US on the number of recorded cases per State"  
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(df, 'id', ['Frequency','state'])
).properties(
    width=500,
    height=300
).add_selection(click).project(
    type='albersUsa'
)

bars = (
    alt.Chart(
        df.nlargest(15, 'Frequency'),
        title='Top 15 states by population').mark_bar().encode(
    x='Frequency',
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    color='Frequency',
    y=alt.Y('state', sort='x'))
.add_selection(click))

st.write(displayUSMap & bars)    

#Reference for Interaction: https://stackoverflow.com/questions/63751130/altair-choropleth-map-color-highlight-based-on-line-chart-selection

# Plotting the Offender Race vs Victim's Hate Crime 
df_HeatMap = df_HeatMap.dropna()
df_HeatMap['VictimHateCrime'] = pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Indian"), 'Anti-Indian',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Arab"), 'Anti-Arab',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Asian"), 'Anti-Asian',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Bisexual"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Gay"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Gender"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Heterosexual"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Lesbian"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Black"), 'Anti-Black',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Islamic"), 'Anti-Islamic',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Female"), 'Anti-sexism',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Male"), 'Anti-sexism',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Hispanic"), 'Anti-hispanic',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Jewish"), 'Anti-Jewish' 
                                , 'Multiple Groups'))))))))))))))
heatmap1 = alt.Chart(df_HeatMap).mark_rect(
    tooltip=True
).encode(
    alt.X("VictimHateCrime", scale=alt.Scale(zero=False), axis=alt.Axis(labelAngle=-0), title='Victim Hate Crime Type'),
    alt.Y("OFFENDER_RACE", scale=alt.Scale(zero=False), title='Offender Race'),
    alt.Color("count():Q", title = "No. of cases")
).properties(
    width=1000,
    height=300,
    title = "Frequency of attacks per Offender's race to Victim's Hate Crime Type"
).configure_title(
    fontSize=20
).configure_axis(
    titleFontSize=18    
)
st.write(heatmap1)

    
 #Clustering   
st.header("Clustering on Hate Crimes")
selection = st.multiselect("Select your features",options = ['Bias','Crime','Location','Offender Race','Victim Type'])

#make selection for clustering
if selection:
    cluster = plot_cluster(selection)
    st.write(cluster)



#Feature Importance
st.header("Feature Importance")
def load_features_final(name):
    return pd.read_csv(name)

df_features_final = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")
# df_hate_crime_data = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")
df_hate_crime_data = df_hate
df_hate_crime_data = df_hate_crime_data.rename(columns={'state_abbr': 'STATE_ABBR'})

state = st.text_input("Enter 2 letter state abbreviation")
#City Visualization
df = df_features_final
df1 = df_hate_crime_data

#displaying the entire united states timeline
US_year = df1.groupby("DATA_YEAR").size()
US_year = US_year.to_frame()
US_year= US_year.reset_index()
US_year.columns = US_year.columns.astype(str)
US_year['DATA_YEAR'] = US_year.DATA_YEAR.astype(str)
lines_state = alt.Chart(US_year).mark_line().encode(
x=alt.X('DATA_YEAR' , scale = alt.Scale(zero=False), title = 'Year', axis=alt.Axis(labelAngle=-0)),
y=alt.Y('0', scale = alt.Scale(zero=False) ,  title =' Number of Crimes')
).properties(
width=1000,
height=300,
title = "Timeline showing the number of  hate crimes in the United States " )

st.header("Visualization showing number of hate crimes over past 28 years in the United States of America")
st.write(lines_state)


#Pie Chart 
if state:
    alt.data_transformers.disable_max_rows()
    # df1 = df_city
    # df1 = pd.read_csv("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/city_data.csv")
    df2 = df_city.pivot_table(index=['state_abbr','metric_name'],values = 'est',aggfunc=np.mean).reset_index()

    i = 0
    value = []
    while i<df2.shape[0]:
        #State name on click
        if df2.iloc[i]['state_abbr']==state:
            if df2.iloc[i]['metric_name']!='Violent crime':
                value.append(df2.iloc[i]['est'])
        i+=1    
    # st.write(value)
    source = pd.DataFrame({"Factors": ['High school completion', 'Income Inequality', 'Life expectancy', 'Neighborhood racial/ethnic segregation', 'Racial/ethnic diversity', 'Unemployment - annual, neighborhood-level', 'Uninsured'], "value": value})
    pie = alt.Chart(source).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="Factors", type="nominal"),
        tooltip = [alt.Tooltip('value')]
    ).properties(
        title = "Well-Being Factors Distribution for the State"
    )
    st.write(pie)


    #selecting only the cities in the selected state
    df.drop(df[df['STATE_ABBR'] != state].index, inplace = True)
    df1.drop(df1[df1['STATE_ABBR'] != state].index, inplace = True)
    
    #select the state for which you want to see the timeline
    state_year = df1.groupby("DATA_YEAR").size()
    state_year = state_year.to_frame()
    state_year = state_year.reset_index()
    state_year.columns = state_year.columns.astype(str)
    state_year['DATA_YEAR'] = state_year.DATA_YEAR.astype(str)

    #st.write(state_year)
    lines_state = alt.Chart(state_year).mark_line().encode(
    x=alt.X('DATA_YEAR' , scale = alt.Scale(zero=False), title = 'Year', axis=alt.Axis(labelAngle=-0)),
    y=alt.Y('0', scale = alt.Scale(zero=False) ,  title =' Number of Crimes')
    ).properties(
    width=1000,
    height=300,
    title = "Timeline showing the number of  hate crimes in  " + state)

    st.header("Visualization showing number of hate crimes over past 28 years in a state")
    st.write(lines_state)

    #number of crimes in various cities of a state
    crime = df1.groupby("PUB_AGENCY_NAME").size()
    no_crimes = crime.to_frame()
    no_crimes.reset_index()
    no_crimes.columns = ['no_of_crimes']
    no_crimes = no_crimes.reset_index()
    no_crimes = no_crimes.rename({'PUB_AGENCY_NAME': 'City'}, axis=1)
    #st.write(no_crimes.head())

    hist = alt.Chart(no_crimes).mark_bar().encode(
    x= alt.X('City', title = 'City', axis=alt.Axis(labelAngle=-0), sort = "-y") , 
    y= alt.Y('no_of_crimes', title = 'Number of Hate Crimes')
    ).properties(
    width=1000,
    height=800,
    title = "Number of crimes in the cities of " + state
    ).transform_window(
        rank='rank(no_of_crimes)',
        sort=[alt.SortField('no_of_crimes', order='descending')]
    ).transform_filter(
    alt.datum.rank <= 10)
        
    st.header("Visualizing number of  Hate crimes in a state")
    st.write(hist)

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
            x= alt.X('X', title = 'metrics', axis=alt.Axis(labelAngle=-0)) , 
            y= alt.Y('Y', title = 'est')
        ).properties(
            width=800,
            height=800,
            title = "Well-Being Factors Distribution for the City of " + city)
            
        st.header("Visualizing well-being factors of a city")
        st.write(hist)
        
        
        
        df1.drop(df1[df1['PUB_AGENCY_NAME'] != city].index, inplace = True)
        year = df1.groupby("DATA_YEAR").size()
        year = year.to_frame()
        year = year.reset_index()
        year.columns = year.columns.astype(str)
        year['DATA_YEAR'] = year.DATA_YEAR.astype(str)
        
        #st.write(year)

        lines = alt.Chart(year).mark_line().encode(
        x=alt.X('DATA_YEAR' , scale = alt.Scale(zero=False), title = 'Year', axis=alt.Axis(labelAngle=-0)),
        y=alt.Y('0', scale = alt.Scale(zero=False) ,  title =' Number of Crimes')
        ).properties(
        width=800,
        height=300,
        title = "Timeline showing the number of hate crimes in  " + city)
        
        st.header("Visualization showing number of hate crimes over past 28 years in a city")
        st.write(lines)
        
        
    else:
        st.write("Please choose a city from the list provided")
    

    
