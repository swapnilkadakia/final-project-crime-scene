import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import cleanup as cp
import umap
from vega_datasets import data
from sklearn.cluster import DBSCAN
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import set_config 
from matplotlib.pyplot import figure


def load_features_final(name):
    return pd.read_csv(name)

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
    
# st.title("Hate Crimes in the United States")

#Formatting
with st.sidebar:
    choose = option_menu("Hate Crimes in the US", ["Home","Exploratory Data Analysis","Hate Crime Distribution", "Clustering","Feature Importance" ,"Exploring States & Cities"],
                         icons=['house','table', 'map', 'circle','bar-chart-line' ,'building'],
                         menu_icon="app", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#000000"},
        "icon": {"color": "blue", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

with st.spinner(text="Loading data..."):
    df_hate,df_city = load_data()

if choose == 'Home':
    st.title("Home")
    st.write("In 2018, on the morning of October 27, Robert Bowers entered the Tree of Life Synagogue in Pittsburgh, PA, yelled “All Jews must die,” and opened fire on the congregants. He was armed with an assault rifle and several handguns, and killed eleven congregants and wounded six others, four of whom are police officers. This was one of the deadliest attacks on the Jewish Community in the United States.")
    # image = Image.open('https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/hate_crime_shooting.png')
    st.image("https://media-cldnry.s-nbcnews.com/image/upload/newscms/2019_35/2624281/181030-pittsburgh-synagogue-mn-1150.jpg")
    st.write("This is just not one isolated incident, Tens of thousands are victims of hate crimes each year. We want to investigate the underlying tendency of hate crimes from 1991 to 2020. To further glean a comprehensive understanding, we will explore specific patterns pertaining to an offender's bias against a race or ethnicity, religion, disability, sexual orientation, gender, or gender identity. We seek to portray the shift in patterns and the factors behind the shifts through each category of incidents. Therefore, we seek to address the question:")
    st.write("How have socioeconomic factors, particularly human well being, influenced hate crime patterns in the United States over the years?")
    st.write("The ultimate goal of our project is to make this world a safer place where every individual can express themselves and cultivate an unbiased society. We plan to use the feature importance of a predictive model to determine the relevant features/demographics which most influence the number of hate crimes in a region. With this, we can uncover the patterns of what some activists refer to as the 'Hate Crime Epidemic' in the United States.")
elif choose == 'Exploratory Data Analysis':
    st.title("Exploratory Data Analysis")

    st.subheader("ABOUT THE DATA:")
    st.markdown("The basis of answering this data science problem is through the Hate_Crime dataset provided by the FBI. The dataset concentrates on hate crimes in different cities across the United States from 1991 to 2020. Some of the features included in this dataset are incident date, location, offender race, offense name. There are about 200,000 rows in the dataset, which gives us a good amount of information to work with and draw conclusions. Furthermore, To acquire a better understanding of what variables contribute to a rise in hate crime in a community, we need to know its demographics and other socioeconomic characteristics. We are using the data provided by 'https://www.cityhealthdashboard.com/' to perform analysis and create visualizations. The dataset downloaded has a variety of metrics included such as Absenteeism , Broadband Connection, Breast Cancer Deaths, COVID Local Risk Index , Cardiovascular Disease Deaths which do not impact the hate crimes in the city and thus we have filtered those out. Some cities in the dataset had multiple instances with the metric values for different groups such as gender, race and total population. We have considered only the total population of the city for our metric evaluation.")
    col1, col2 = st.columns(2)
    # st.write("FBI Hate Crimes Dataset" | "FBI Hate Crimes Dataset")
    col1.subheader("FBI Hate Crimes Dataset")
    col1.write(df_hate.head())
    col2.subheader("Cities Dataset")
    col2.write(df_city.head())

    option = st.selectbox("Select your features",options=['Hate Crimes',"Wellbeing of Cities"])

    if option == 'Hate Crimes':

        st.subheader("Total Recorded Crime Count per State")
        cities = alt.Chart(df_hate.head(1000)).mark_bar().encode(
        x=alt.X("STATE_NAME",  sort="-y"),
        y=alt.Y("count()")
        ).properties(
        width=400,
        height=600,
        title = "Top Cities with Reported Hate Crimes").configure_mark(color='#F1B46D')
        st.write(cities)
        st.write("The above bar graph represents the total number of crimes committed per state in the United States. In terms of the crimes recorded from 1990 to 2020, California has the highest crime rate with over 35,000 crimes recorded.")

        # Offender Race 
        st.subheader("Total Recorded Crime Count per Offfender's Race and Victim's Race")
        offenderRace = alt.Chart(df_hate.head(1000)).mark_bar().encode(
            x=alt.X("OFFENDER_RACE",  sort="-y"),
            y=alt.Y("count()")
        ).properties(
        width=300,
        height=800,
        title = "Offender Race").configure_mark(color='#F8EED3')

        # Victim Race 
        bd = df_hate['BIAS_DESC']
        bd= bd.apply(str)
        bd = pd.DataFrame(bd)
        victimRace = alt.Chart(bd.head(1000)).mark_bar().encode(
            x=alt.X("BIAS_DESC",  sort="-y"),
            y=alt.Y("count()")
        ).transform_window(
            rank='rank(BIAS_DESC)'
        ).transform_filter(
            (alt.datum.rank < 1000)
        ).properties(
        width=500,
        height=800,
        title = "Victim Race").configure_mark(color='#31AE9D')

        offend, vict = st.columns(2)
        offend.write(offenderRace)
        vict.write(victimRace)
        st.write("The above bar graphs represents the total number of crimes committed per offender racial group and per victim group in the United States. In terms of the crimes recorded from 1990 to 2020, most of the crimes that were recorded did not specify the offender’s race and were Anti-Black crimes committed against African Americans.")

    else:
        #Plotting High School Completion 
        st.subheader("High School Completion")
        st.write("The above bar graph represents the high school completion rates across the United States. Minnesota has the highest high school completion rate of 98.3.")
        data_final1  = df_city[df_city["metric_name"].isin(["High school completion"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        High_School_Completion_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[85,100]))
        ).properties(
        title = "High School Completion Well-Being Factor across States").configure_mark(color='#C64863')

        st.write(High_School_Completion_bar)

        #Plotting Life expectancy 
        st.subheader("Life expectancy")
        st.write("The above bar graph represents the life expectancy rates across the United States. California has the highest life expectancy rate of 85.5.")
        data_final1  = df_city[df_city["metric_name"].isin(["Life expectancy"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Life_expectancy_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[74,86]))
        ).properties(
        title = "Life expectancy Well-Being Factor across States").configure_mark(color='#E4CDDD')
        
        st.write(Life_expectancy_bar)


        #Plotting Income Inequality 
        st.subheader("Income Inequality ")
        st.write("The above bar graph represents the income inequality rates across the United States. Washington has the highest income inequality rate of around 56.")
        data_final1  = df_city[df_city["metric_name"].isin(["Income Inequality"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Income_Inequality_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,60]))
        ).properties(
        title = "Income Inequality Well-Being Factor across States").configure_mark(color='#726CA8')

        st.write(Income_Inequality_bar)


        #Plotting Neighborhood racial/ethnic segregation
        st.subheader("Neighborhood racial/ethnic segregation")
        st.write("The above bar graph represents the neighborhood racial/ethnic segregation rates across the United States. Illinois has the highest neighborhood racial/ethnic segregation rate of 43.5.")
        data_final1  = df_city[df_city["metric_name"].isin(["Neighborhood racial/ethnic segregation"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Neighborhood_racial_ethnic_segregation_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,45]))
        ).properties(
        title = "Neighborhood racial/ethnic segregation Well-Being Factor across States").configure_mark(color='#D9BAD2')

        st.write(Neighborhood_racial_ethnic_segregation_bar)

        #Plotting Racial/ethnic diversity
        st.subheader("Plotting Racial/ethnic diversity")
        st.write("The above bar graph represents the racial/ethnic diversity rates across the United States. California has the highest racial/ethnic diversity rate of 94.5.")
        data_final1  = df_city[df_city["metric_name"].isin(["Racial/ethnic diversity"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Racial_ethnic_diversity_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[35,100]))
        ).properties(
        title = "Racial/ethnic diversity Well-Being Factor across States").configure_mark(color='#5782A6')

        st.write(Racial_ethnic_diversity_bar)

    
        #Plotting Unemployment Factor
        st.subheader("Plotting Unemployment Factor")
        st.write("The above bar graph represents the unemployment rates across the United States. Michigan has the highest unemployment rate at 19.5.")
        data_final1  = df_city[df_city["metric_name"].isin(["Unemployment - annual, neighborhood-level"])]
        data_final2  = data_final1[data_final1["group_name"].isin(["total population"])]
        data_final3 = data_final2.sort_values(by='est', ascending=False)
        data_final3.head(1000)

        Unemployment_Factor_bar = alt.Chart(data_final3).mark_bar(clip=True).encode(
            x=alt.X('state_abbr', sort = None),
            y=alt.Y('est', scale=alt.Scale(domain=[0,20]))
        ).properties(
        title = "Unemployment Factor Well-Being Factor across States").configure_mark(color='#34a230')

        st.write(Unemployment_Factor_bar)

elif choose == "Hate Crime Distribution":
    #US MAP
    st.title("Hate Crime Distribution")    

    #displaying the entire united states timeline
    df_year = df_hate['DATA_YEAR'].copy().reset_index()
    # df_year = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")

    
    st.subheader("Visualization showing number of hate crimes over past 28 years in the United States of America")
    st.write("This visualization depicts the evolution of hate crime statistics in the United States over the last 20 years. The x-axis represents every year from 1991 to 2020, and the y-axis is the total number of cases. The sharp spikes and drops in hate crimes can be linked to major events that occurred in the United States. Riots, presidential elections, policy decisions, and global pandemics are some of these events. The most concerning aspect of this graph is the never-before-seen surge in the number of incidents after the Covid-19 Pandemic began in 2019. ")
   
    year_range = st.slider('DATA_YEAR',
                        min_value=int(df_year['DATA_YEAR'].min()),
                        max_value=int(df_year['DATA_YEAR'].max()),
                        value=(int(df_year['DATA_YEAR'].min()), int(df_year['DATA_YEAR'].max())))

    df_year.drop(df_year[df_year['DATA_YEAR'] <= year_range[0]-1].index, inplace = True)
    df_year.drop(df_year[df_year['DATA_YEAR'] >= year_range[1]+1].index, inplace = True)


    lines_state = alt.Chart(df_year).mark_line().encode(
    x=alt.X('DATA_YEAR:N' , scale = alt.Scale(zero=False), title = 'Year', axis=alt.Axis(labelAngle=-0)),
    y=alt.Y('count()', scale = alt.Scale(zero=False) ,  title =' Number of Crimes')
    ).properties(
    width=1000,
    height=300,
    title = "Timeline showing the number of  hate crimes in the United States "
    ).interactive(
    bind_y = False
    )

    #df_year['DATA_YEAR'] = df_year.DATA_YEAR.astype(str)
    #df1['DATA_YEAR'] = df1.DATA_YEAR.astype(str)

    #lines_state.interactive(bind_y=False)
    st.write(lines_state)


    # Importing Data
    df =pd.read_csv("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")
    # df = df_hate
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
    st.write("The visualization depicts a Choropleth Map of the United States, based on the total number of cases in each state and a Bar Graph of the top 15 states with the highest number of hate crimes. With selection of states on the US map, the same states get highlighted on the bar graph. California has the highest number of hate crime cases which is followed by New York.")
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
            title='Top 15 states by Frequency of Hate Crime').mark_bar().encode(
        x='Frequency',
        opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
        color='Frequency',
        y=alt.Y('state', sort='x'))
    .add_selection(click))

    st.write(displayUSMap & bars)    

    #Reference for Interaction: https://stackoverflow.com/questions/63751130/altair-choropleth-map-color-highlight-based-on-line-chart-selection

    # Plotting the Offender Race vs Victim's Hate Crime 
    st.subheader("Correlation of Offender's Race to Victim's Hate Crime Type")
    st.write("The visualization depicts the Victim's Ethnicity is on the x-axis, while the Offender's Race is shown on the y-axis in this heat map. The graph shows that most hate crimes are committed against African Americans, followed by attacks on the LGBTQ+ community, as corroborated by clustering. We also see that the most common offender is white, which may be since white people make up the bulk of the population in the United States. Crimes against African Americans committed by white offenders are the most serious type of hate crime, according to the below visualization.")
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

elif choose =='Clustering':
    #Clustering   
    st.header("Clustering")
    st.write("In this section, we will be exploring clustering techniques on features such as Bias, Location, Offender Race, Crime, and Victim Type. DBSCAN and DensMAP algorithms were used to reduce the features and cluster the data.")    
    st.write("Clustering helps us visualize our based on how the data is grouped according to certain characteristics. We can see the impact of these characteristics by iteratively adding them to our algorithm. The groupings produced allow us to see how similar characteristics in incidents are common in certain crimes. With clustering, we can direct our analysis to how crimes have occurred in the past and help inform us of where our further research should take us.")

    selection = st.multiselect("Select your features",options = ['Bias','Crime','Location','Offender Race','Victim Type'])

    #make selection for clustering
    if selection:
        cluster = plot_cluster(selection)
        st.write(cluster)
        st.write("The above visualization displays clusters based on the user-selected features. The addition of more features creates sparse clusters.") 
elif choose =='Feature Importance':
    st.title("Feature Importance")
    df_features_final1 = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")

    st.write("In this section, we will be exploring the important features that impact hate crimes in US states. This graph can be customized to be displayed for the states and the features the user is interested in.")

    #Feature exploration
    st.header('Feature Exploration')

    states = df_features_final1['STATE_ABBR'].unique()
    features = df_features_final1.columns[2:9]

    
    feat_selection = st.multiselect("Select the features you want to explore", options = features)

    if feat_selection:
     #running ML for feature importance
        df_features_final1 = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")
        df_m2 = df_features_final1

        Y_m2 = df_m2["no_of_crimes"]  
        X_m2 = df_m2[feat_selection]
            # define the model
        model = RandomForestRegressor()
            # fit the model
        model.fit(X_m2, Y_m2)
            # get importance
        importance = model.feature_importances_

            # summarize feature importance
        for i in range(len(importance)):
            print(X_m2.columns[i] + " : "+ str(importance[i].round(6)))
                

        figure(figsize=(16, 10), dpi=80)
                
        # plot feature importance
        plt.xticks(rotation=0)
        plt.bar([x for x in X_m2.columns], importance, color = 'orange')

        st.pyplot(plt)

        state_selection = st.multiselect("Select the states you want to explore", options = states)

        if state_selection:
            tansposed = df_features_final1[df_features_final1['STATE_ABBR'].isin(state_selection)][feat_selection].T.reset_index()

            chartList = []

            for state in state_selection:
                tansposed = df_features_final1[df_features_final1['STATE_ABBR'].isin(state_selection)][feat_selection+['no_of_crimes']].T.reset_index()
                tansposed['mean'] = tansposed.mean(axis=1)
                exploration_chart = alt.Chart(tansposed).mark_bar().encode(
                        alt.X("mean:Q", scale = alt.Scale(zero=True)),
                        alt.Y("index"),
                        alt.Color("index:N")
                    ).properties(
                width=500,
                height=200,
                title = "'{}' features".format(state))

                st.write(exploration_chart)
    


elif choose == "Exploring States & Cities":

    #Exploring
    st.header("Exploring States & Cities")
    st.write("In this section, we will be exploring the states of US. We will first explore the distribution of well-being factors in the state using a pie chart. Next, we will represent the overall hate crime cases in the state over the past 3 decades in the form of a line graph. This is followed by a bar graph that compares the hate crime rates across the various cities in the state.")

    df_features_final = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")
    # df_hate_crime_data = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")
    df_hate_crime_data = df_hate
    df_hate_crime_data = df_hate_crime_data.rename(columns={'state_abbr': 'STATE_ABBR'})

    state = st.text_input("Enter 2 letter state abbreviation")
    #City Visualization
    df = df_features_final
    df1 = df_hate_crime_data


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

        st.write("The line graph and bar graph are interlinked. Selection of a particular time period in the line graph displays the crime rates per city for those years.") 

        #Hist and bar 
        df1 = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/hate_crime.csv")
        df1.drop(df1[df1['STATE_ABBR'] != state].index, inplace = True)
        df1['count'] = df1.groupby('PUB_AGENCY_NAME')['PUB_AGENCY_NAME'].transform('count')
        # df_city = df1
        df1['DATA_YEAR'] = df1.DATA_YEAR.astype(str)
         

        brush = alt.selection_interval()
        #st.write(state_year)
        lines_state = alt.Chart(df1).mark_line().encode(
        x=alt.X('DATA_YEAR' , scale = alt.Scale(zero=False), title = 'Year', axis=alt.Axis(labelAngle=-0)),
        y=alt.Y('count(DATA_YEAR)', scale = alt.Scale(zero=False) ,  title =' Number of Crimes'),
        color= alt.condition(brush, alt.value("red"), alt.value("grey"))
        ).properties(
        width=1000,
        height=300,
        title = "Timeline showing the number of  hate crimes in  " + state
        ).add_selection(
        brush)

        st.header("Visualization showing number of hate crimes over past 28 years in a state")

        hist = alt.Chart(df1).mark_bar().encode(
        x= alt.X('PUB_AGENCY_NAME', title = 'City', axis=alt.Axis(labelAngle=-0), sort = "-y") , 
        y= alt.Y('count()', title = 'Number of Hate Crimes'), 
        color = alt.Color('TOTAL_OFFENDER_COUNT'),
        tooltip = [alt.Tooltip('LOCATION_NAME:N'),
                alt.Tooltip('BIAS_DESC:N'),
                alt.Tooltip('OFFENSE_NAME:N'),
                ]
        ).properties(
        width=1200,
        height=800,
        title = "Number of crimes in the cities of " + state
        ).transform_window(
        rank= 'dense_rank(count())',
        sort = [alt.SortField('count', order = 'descending')]
        ).transform_filter(
        alt.datum.rank <= 10,
        ).transform_filter(
        brush)
        selection = alt.selection_interval(bind='scales')
        st.write(lines_state & hist)

        # #selecting only the cities in the selected state
        df.drop(df[df['STATE_ABBR'] != state].index, inplace = True)

        st.write("In this section, we will be exploring the cities of the US. We will explore the overall hate crimes in the city over the past 3 decades.")
        st.write("Enter one of the following cities in " + state)
        st.write(df['City'])

        #select the city for which you want to see the factors
        city = st.text_input("Enter city name")
        if( city in df.values  ):
            df.drop(df[df['City'] != city].index, inplace = True)
            print(df.head())
            
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
    

    
