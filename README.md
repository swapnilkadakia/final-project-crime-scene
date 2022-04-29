# CMU Interactive Data Science Final Project

* **Online URL**: https://share.streamlit.io/cmu-ids-2022/final-project-crime-scene/main/code/hate-crimes.py
* **Report.md URL**: https://github.com/CMU-IDS-2022/final-project-crime-scene/blob/main/Report.md
* **Video URL**: https://youtu.be/QRqiKuS_Ymc
* **Team members**:
  * Contact person: swapnilk@andrew.cmu.edu
  * aaronho@andrew.cmu.edu
  * nninan2@andrew.cmu.edu
  * neeman@andrew.cmu.edu

## Abstract

The data science problem we aim to address is how human well-being in cities and states influences hate crime patterns over the years. Our approach to this problem is to address these issues and offer a broader exploration into the behaviors, demographics, and other defining features behind the causes of hate crimes. Our approach first provides a high-level overview of the prevalence of hate crimes in the U.S. from 1990 to 2020. Pre-processing the data, clustering, correlation, feature importance, the timeline of hate crimes, hate crime distribution across the United States, variables contributing to the hate crime rate of each state, and analyzing hate crime statistics for cities in each state are all steps in the process. This analysis aligns with our expectation of multiple factors resulting in hate crimes. Our efforts will attempt to simulate that as much as possible. Users can explore the patterns in demographics and behaviors of a particular state and the prevalence of hate crimes in that state. The expectation is that the features associated with a state (i.e., education, racial diversity, income) would help explain patterns in hate crimes in that state. In exploring this relation, we aim to clarify what influences hate crimes and inform policy actions and decisions to address these crimes in cities and states. The further study generated from the conclusions of these results would be to explore the most defining features and focus research on how these features contribute to biases and crime.


## Work distribution

Team division:
- Feature Importance – Swapnil
- Clustering Techniques – Aaron
- Project visualization – Neema, Swapnil, Natasha, Aaron
- PPT – Neema
- Report – Natasha, Neema, Aaron, Swapnil
- Video - Aaron

## Commentary
During the project process we were fortunate to be able to work with a dataset that was comprehensive and covered a substantial portion of the years we wanted to look at for our project. This dataset, FBI Hate Crimes, formed the basis of our analysis and allowed us to combine it with our datasets such as our cities dataset regarding demographic information. Because it was a clean csv file, we didn't need to do too much data cleanup besides engineer our features for analysis. The main challenge we ran into was consolidating all our code as well as reducing the size of the data we are using. Streamlit ran into problems with charts utilizing a larger sample, so we therefore had to reduce the features in order to allow it to still compile in a reasonable amount of time. Overall, we are pleased with how our application turned out and we are delighted to present some of the functionality that we incorporated into our application.

## Deliverables

### Proposal

- [X] The URL at the top of this readme needs to point to your application online. It should also list the names of the team members.
- [X] A completed [proposal](Proposal.md). Each student should submit the URL that points to this file in their github repo on Canvas.

### Sketches

- [X] Develop sketches/prototype of your project.

### Final deliverables

- [X] All code for the project should be in the repo.
- [X] Update the **Online URL** above to point to your deployed project.
- [X] A detailed [project report](Report.md).  Each student should submit the URL that points to this file in their github repo on Canvas.
- [ ] A 5 minute video demonstration.  Upload the video to this github repo and link to it from your report.

### Run Instructions:
1. Click on the Online URL at the top of this file to open the Streamlit application.
2. You should see six tabs on the side, detailing which section to look at with the application. You may explore these sections in any order as you wish. 
3. Certain sections may take longer to show. Please wait about a minute for them to load. 
