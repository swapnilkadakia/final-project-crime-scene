# Final Project Proposal

Team Members:
Aaron Ho,
Neema Nayak,
Natasha Ninan,
Swapnil Kadakia

**GitHub Repo URL**: https://github.com/CMU-IDS-2022/final-project-crime-scene

Question: How have socioeconomic factors, particularly human well-being, influenced hate crime patterns in the United States over the years?

From the start of the pandemic in 2020 to our current situation now, we have seen a rise in hate-related crimes, including but not limited to assault, murder, verbal abuse, and property damage. As a team, we have noticed that these came about due to increasing racial tensions exasperated by the pandemic, which encouraged misplaced blame and subsequent racial violence. However, hate crimes are not a recent phenomenon and extend beyond racial identities. In our research for this project, we have found that hate crimes have predated the start of the century and include violence towards members of the LGBTQ community, various religious groups, ethnicity, persons with disability, and gender. And so, our project will be addressing the data science problem of how human well-being in cities and states influences hate crimes over the years. Given hate crime data at the incident level from 1991 to 2020 compiled by the FBI, our solution will address the impact of demographics such as income, education, and diversity on hate crime, trends in types of hate crimes committed, and suggestions for areas of improvement. We aim to accomplish this through an interactive visualization/application approach.

The basis of answering this data science problem is through the Hate_Crime dataset provided by the FBI. The dataset concentrates on hate crimes in different cities across the United States from 1991 to 2020. Some of the features included in this dataset are incident date, location, offender race, offense name. There are about 200,000 rows in the dataset, which gives us a good amount of information to work with and draw conclusions. We aim to gather data from multiple sources as we progress through the project to support our analysis. Datasets on overall well-being and demographics in the U.S. will help substantiate our insights. 

According to FBI statistics, hate crimes in the United States have reached an all-time high. Through this project, we plan to investigate the factors of each state that directly correlate with the increasing hate crime. One of the major factors that we will be looking at is each state's overall well-being, which includes the state's overall educated population, income rates, health care facilities, and age groups, to name a few. The well-being of each state combined with the crime rates will enable us to understand the distribution of hate crime across U.S. states based on the type of hate crime and the total hate crime registered cases.

The most recent spike in hate crime heralds a new cruel landscape in which targeted attacks against victim groups have resulted in widespread increases in the most violent crimes. We want to investigate the underlying tendency of hate crimes from 1991 to 2020. To further glean a comprehensive understanding, we will explore specific patterns pertaining to an offender's bias against a race or ethnicity, religion, disability, sexual orientation, gender, or gender identity. We seek to portray the shift in patterns and the factors behind the shifts through each category of incidents. 

The ultimate goal of our project is to make this world a safer place where every individual can express themselves and cultivate an unbiased society. We plan to use the feature importance of a predictive model to determine the relevant features/demographics which most influence the number of hate crimes in a region. With this, we can uncover the patterns of what some activists refer to as the "Hate Crime Epidemic" in the United States.


<h1>Sketches and Data Analysis</h1>

<h2>Data Processing</h2>

<h3>Hate Crime Data</h3>

Since the FBI Hate Crime dataset formed the basis for our analysis, we had to undergo substantial data processing on this dataset alone so that it is viable for our visualizations and modeling. In our first processing step, we remove unused features that may be redundant or uninformative for our intended analysis. For example, ‘ORI’ refers to a nine-character identifier for authorized agencies to access Criminal Justice Information (CJI) and ‘ADULT_OFFENDER_COUNT’ refers to the number of offenders out of the total, that are considered to be an adult. In the former, this is not very informative and will not affect our analysis and thus can be removed. In the latter, we removed this feature because we are mainly interested in the overall offender count and therefore this feature will not be used and can be removed. Following the removal of unnecessary features, we ensured that numeric features are set to the proper integer datatype as well as to the proper date datatype. Below is a screenshot of our updated features and their datatypes. 

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/new_features.PNG" width="250" height="250">

Upon further exploration of our dataset, we realized that many of the current categorical features made it difficult to pursue further analysis without further transformation. We adopted to use one hot encoding to transform our data into a binary representation. We wanted to do the same for other columns, but those columns seemed to indicate multilabeling. In these cases, we used the MultiLabelBinarizer from sklearn. This would allow us to give binary representation to multiple labels that a hate crime instance can have in a category. An example of this is shown below for the BIAS_DESC column.

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/mlb.PNG" width="250" height="250">

 Through these transformations, we expect that our features will be more expressive and we can further explore feature relationships in our visualizations. We also hope this will allow us to train predictive models on these categorical features. Our cleaned data for the FBI Hate Crimes dataset is shown below.

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/transformed_dataframe.PNG">

<h3>Wellness Factors of US Cities Data</h3>


From our exploration and data cleanup, we expect our data to be prepared to help address our data science problem. 

<h2>Data Exploration</h2>

<h3>1. Hate Crime Dataset</h3>

<h4>1.1 Total Recorded Crime Count per State</h4>

<img width="468" alt="image" src="https://user-images.githubusercontent.com/90164318/163605570-e3f7ed19-f0eb-47ab-a03a-16eb83c1d63e.png">

The above bar graph represents the total number of crimes committed per state in the United States. In terms of the crimes recorded from 1990 to 2020, California has the highest crime rate with over 35,000 crimes recorded.


<h4>1.2 Total Recorded Crime Count per Offfender's Race and Victim's Race</h4>
<img width="307" alt="image" src="https://user-images.githubusercontent.com/90164318/163626186-1cddcbb2-ce3a-4205-a954-43bfab02e14c.png">

The above bar graphs represents the total number of crimes committed per offender racial group and per victim group in the United States. In terms of the crimes recorded from 1990 to 2020, most of the crimes that were recorded did not specify the offender’s race and were Anti-Black crimes committed against African Americans.

<h4>1.3 Total Recorded Crimes per Year</h4>

<img width="306" alt="image" src="https://user-images.githubusercontent.com/90164318/163608990-fcc41e6b-40cc-4754-887a-b13ed3e507db.png">

The above line graph represents the total number of crimes committed per year from 1990 to 2020 in the United States. The maximum number of crimes were committed in the year 2020 with a value close to 14,000.



<h3>2. Wellness Factors of Cities</h3>

<h4>2.1 High School Completion Factor across the states of the USA</h4>
<img width="193" alt="image" src="https://user-images.githubusercontent.com/90164318/163625330-d07e48e0-6f80-47f0-a332-2a4beeb85df8.png">
The above bar graph represents the high school completion rates across the United States. Minnesota has the highest high school completion rate of 98.3.

<h4>2.2 Life Expectancy Factor across the states of the USA</h4>
<img width="203" alt="image" src="https://user-images.githubusercontent.com/90164318/163625382-cd3309d4-4fdb-4403-b0d3-1db34e3c091c.png">
The above bar graph represents the life expectancy rates across the United States. California has the highest life expectancy rate of 85.5. 

<h4>2.3 Income Inequality Factor across the states of the USA</h4>
<img width="232" alt="image" src="https://user-images.githubusercontent.com/90164318/163625428-48795423-8776-4076-8d6c-5fa863f7f93e.png">
The above bar graph represents the income inequality rates across the United States. Washington has the highest income inequality rate of around 56. 

<h4>2.4	Neighborhood racial/ethnic segregation Factor across the states of the USA</h4>
<img width="224" alt="image" src="https://user-images.githubusercontent.com/90164318/163625484-d0e99aad-1719-4f98-b6f9-23ca7ff2a9e0.png">
The above bar graph represents the neighborhood racial/ethnic segregation rates across the United States. Illinois has the highest neighborhood racial/ethnic segregation rate of 43.5. 

<h4>2.5	Racial/Ethnic Diversity Factor across the states of the USA</h4>
<img width="196" alt="image" src="https://user-images.githubusercontent.com/90164318/163625643-d917bedc-1d43-4be4-a62e-3b15e1c57731.png">
The above bar graph represents the racial/ethnic diversity rates across the United States. California has the highest racial/ethnic diversity rate of 94.5.

<h4>2.6	Unemployment Factor across the states of the USA</h4>
<img width="186" alt="image" src="https://user-images.githubusercontent.com/90164318/163625674-05c59518-c32a-4b59-b9b1-e0361f070078.png">
The above bar graph represents the unemployment rates across the United States. Michigan has the highest unemployment rate at 19.5. 

<h4>2.7	Voilet Crimes Factor across the states of the USA</h4>
<img width="215" alt="image" src="https://user-images.githubusercontent.com/90164318/163625688-bb50bf3a-25d0-4dcf-a868-6205a1b1f0c2.png">
The above bar graph represents violent crime rates across the United States. Michigan has the highest violent crime of 1,920. 

<h4>2.8	Uninsured Factor across the states of the USA</h4>
<img width="207" alt="image" src="https://user-images.githubusercontent.com/90164318/163625715-fea94f5a-5da4-4691-aa58-f452d0cfd670.png">
The above bar graph represents the uninsured rates across the United States. Texas has the highest uninsured rate at 35. 

<h4>2.9	Binge Drinking across the states of the USA</h4>
<img width="216" alt="image" src="https://user-images.githubusercontent.com/90164318/163625736-5db068c0-2928-4703-91b0-d0bc148982e4.png">
The above bar graph represents the binge drinking rates across the United States. Iowa has the highest binge drinking rate at 26.5. 

<h4>2.10	Housing Costs across the states of the USA</h4>
<img width="211" alt="image" src="https://user-images.githubusercontent.com/90164318/163625766-a8dcc303-12d6-4be8-a596-d7080fbdf537.png">
The above bar graph represents the housing costs across the United States. New Jersey has the highest housing costs rate at 59.5. 


<h2>System Design</h2>
  
How will you display your data? What types of interactions will you support? 
Provide some sketches that you have for the system design.

The data will be displayed in the form of multiple charts which include bar graphs, line graphs, pie charts, bubble charts, and heat maps. Some of the sketches to represent the data are shown below:

<h4>1.	Heat Map of the United States based on the number of recorded cases</h4>
<img width="396" alt="image" src="https://user-images.githubusercontent.com/90164318/163602909-d5fc3aa4-e40d-429b-82b4-b928fcc70bdf.png">
The above heat maps highlight the number of recorded cases. California and New Jersey have the highest recorded crime cases.

Reference: https://www.google.com/url?sa=i&url=https%3A%2F%2Frockcontent.com%2Fblog%2Fyou-are-here-using-maps-in-data-visualization%2F&psig=AOvVaw3Fz71blhi95An1Fhd5EMOm&ust=1650139925547000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCOCgvLLwlvcCFQAAAAAdAAAAABAD


<h4>2.	Bubble Chart for Assault Types</h4>
<img width="285" alt="image" src="https://user-images.githubusercontent.com/90164318/163602946-57f3d1ae-7e95-47e3-8438-2a4eb353f2cc.png">
The above bubble chart highlights the distribution of types of crimes committed. Intimidation is the highest crime type committed.

Reference: https://www.google.com/url?sa=i&url=https%3A%2F%2Finterworks.com%2Fblog%2Fccapitula%2F2015%2F01%2F06%2Ftableau-essentials-chart-types-packed-bubbles%2F&psig=AOvVaw0SEjKXDy0Ud6hD7iCLEh2w&ust=1650139973780000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCPDl7szwlvcCFQAAAAAdAAAAABAD

<h4>3.	Heat Map mapping the ethnicity of the offender to the ethnicity of the victim they attacked</h4>
<img width="353" alt="image" src="https://user-images.githubusercontent.com/90164318/163602992-860f2357-b507-4364-9f10-b4451d54b497.png">
The above heat map shows the mapping of the race of the offender and the race of the victim. 

Reference: https://www.google.com/url?sa=i&url=https%3A%2F%2Fkryotech.co.uk%2Fdata-visualization-for-beginners-part-3%2F&psig=AOvVaw1wTyzOPBppEPaZiTVrfkxS&ust=1650140066991000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCMCCt_nwlvcCFQAAAAAdAAAAABAD


<h4>4.	Interaction</h4>
Some of the interactions that will be included are a dropdown feature to select the state and display a pie chart that shows the distribution of various well-being factors of that state, and a dropdown feature to select a well-being factor and show its distribution among all the US states. 

<h5>4.1	Dropdown of state name displaying well-being factors distribution</h5>
<img width="408" alt="image" src="https://user-images.githubusercontent.com/90164318/163628826-2abe8727-5af8-4e27-b067-eab63602b8b1.png">

<h5>4.2	Dropdown of well-being factors displaying state distribution</h5>
<img width="272" alt="image" src="https://user-images.githubusercontent.com/90164318/163628836-8a46b1a2-b525-4cac-a628-3fefea2b4bff.png">

<h5>4.3	Checkbox to select victim race and display recorded crime rates across the years</h5>
<img width="369" alt="image" src="https://user-images.githubusercontent.com/90164318/163628847-c2bf24cb-2ecf-4d55-aae6-5c9f254fc2f7.png">


<h4>5. </h4>
One of the facets of the data science problem that we aimed to answer through our exploration and analysis was to determine which specific features were more influential in causing hate crimes. We aimed to look demographic information as well as circumstances surrounding who, where and how a crime was committed. Our purpose was to look for patterns or latent structures that can inform our understanding of hate crimes. Therefore, we intend to explore this through some dimensionality reduction and clustering techniques so that we can determine any informative features. Using GMM clustering grants us more flexibility regarding how patterns or clusters are formed and shaped as well being more relaxed in the how data points are assigned. We imagine our clustering to look similar to the sketch below with differing colors to indicate label assignments. To reduce the overall number of features needed to present this clustering, we also decided to use a drop-down menu so that users can select the type of features they want to explore in relation to hate crime occurrences.
<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/clustering_sketch.jpg" width="400" height="300">
