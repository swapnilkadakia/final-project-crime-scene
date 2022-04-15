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

Do you have to do substantial data cleanup?
What quantities do you plan to derive from your data?
How will data processing be implemented? 
Show some screenshots of your data to demonstrate you have explored it.

Since the FBI Hate Crime dataset formed the basis for our analysis, we had to undergo substantial data processing on this dataset alone so that it is viable for our visualizations and modeling. In our first processing step, we remove unused features that may be redundant or uninformative for our intended analysis. For example, ‘ORI’ refers to a nine-character identifier for authorized agencies to access Criminal Justice Information (CJI) and ‘ADULT_OFFENDER_COUNT’ refers to the number of offenders out of the total, that are considered to be an adult. In the former, this is not very informative and will not affect our analysis and thus can be removed. In the latter, we removed this feature because we are mainly interested in the overall offender count and therefore this feature will not be used and can be removed. Following the removal of unnecessary features, we ensured that numeric features are set to the proper integer datatype as well as to the proper date datatype. Below is a screenshot of our updated features and their datatypes. 

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/new_features.PNG" width="250" height="250">

Upon further exploration of our dataset, we realized that many of the current categorical features made it difficult to pursue further analysis without further transformation. We adopted to use one hot encoding to transform our data into a binary representation. We wanted to do the same for other columns, but those columns seemed to indicate multilabeling. In these cases, we used the MultiLabelBinarizer from sklearn. This would allow us to give binary representation to multiple labels that a hate crime instance can have in a category. An example of this is shown below for the BIAS_DESC column.

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/mlb.PNG" width="250" height="250">

 Through these transformations, we expect that our features will be more expressive and we can further explore feature relationships in our visualizations. We also hope this will allow us to train predictive models on these categorical features. Our cleaned data for the FBI Hate Crimes dataset is shown below.

<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/transformed_dataframe.PNG">

From our exploration and data cleanup, we expect our data to be prepared to help address our data science problem. 

<h2>System Design</h2>
  
How will you display your data? What types of interactions will you support? 
Provide some sketches that you have for the system design.

One of the facets of the data science problem that we aimed to answer through our exploration and analysis was to determine which specific features were more influential in causing hate crimes. We aimed to look demographic information as well as circumstances surrounding who, where and how a crime was committted. Our purpose was to look for patterns or latent structures that can inform our understanding of hate crimes. Therefore, we intend to explore this through some dimensionality reduction and clustering techniques so that we can determine any informative features. Using GMM clustering grants us more flexibility regarding how patterns or clusters are formed and shaped as well being more relaxed in the how data points are assigned. We imagine our clustering to look similar to the sketch below with differing colors to indicate label assignments. To reduce the overall number of features needed to present this clustering, we also decided to use a drop down menu so that users can select the type of features they want to explore in relation to hate crime occurrences. 
<img src="https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/images/clustering_sketch.jpg" width="250" height="250">
