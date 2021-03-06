<h1>About The Project</h1>
<h2>Notable Contents</h2>
<ul>
    <li>Code folder: Contains code for the application as well as several jupyter notebooks.</li>
    <li>EDA Notebook: Located in the Code folder.</li>
    <li>Pickles Folder: This contains resources saved in bytes for use by our code.</li>
    <li>Data Folder: This contains .csv files for use by our code.</li>
    <li>Presentation.pdf: a pdf version of the powerpoint presentation used to introduce the model</li>
</ul>
<h2>Problem Statement</h2>
<p>This project sets out to do three things. Firstly to take two subreddits about related video games and 
build a predictive model capable of determining which of the two games a given text refers to. Secondly to establish which type of model is better suited to the task. Finally the project seeks to create an easy to use user interface with which an end user can supply a text and have the machine give a prediction.</p>
<p>The question may be posed of what use is the model. While in truth there may not be a great need to predict between two video games this project serves as a proof of concept. A larger model could be created with data from several subreddits and tied with a web crawler it could be used to find online content both in and out of reddit related to a given subject with great accuracy. This also serves as my recommended next step after this project.</p>
<h2>Methodology</h2>
<p>All models used for this project used a simple bag of words array for their training data. Three different models were used in the course of this project. The first was a simple linear regression model trained on the titles of roughly 15,000 reddit posts. This model required tuning to prevent it from overfitting. Eventually the solution to this was to set the minimum occurrences of words to be used to 100 and the maximum occurrences to 400. The second model was in fact identical to the first with the exception being that it was trained on the text of the posts rather than the titles. This proved to be a more reliable model and so took the place of the first.</p>
<p>The third and final model used was a random forest. a grid search object was used to tune the features on this model which ended up with the following: 
    <ul>
        <li> Max depth of each decision tree: 50,</li> 
        <li> Min samples per leaf: 15</li> 
        <li>Min samples per split: 15</li> 
    </ul>
Interestingly this ended up creating a model with the exact same performance as the decided upon linear regression model. In the bellow confusion matrices 1 stands in for a prediction of Dark Souls and 0 for Sekiro Shadows Die Twice. 
</p>
<img src="http://www.andrewgossage.net/wp-content/uploads/2021/09/random-forest-matrix.png"></img>
<img src="http://www.andrewgossage.net/wp-content/uploads/2021/09/logistic-matrix.png"></img>
<p>Although the predictions on the test data were identical the importance given to each individual word by the models was not. That being said 75% of the top 250 words for both models were shared, however their exact places did differ. To investigate this correlation I used a linear regression model to try to predict the importance placed on a word by the logistic model using only the importance placed on the same word by the random forest model. This regressive model was able to predict importance within 1/3 of the standard deviation of the logistic models importance. As a note the importance metric use for the logistic regression model was simply the absolute value of each of its coefficients.</p>
<p>Given the identical performance of these two model I have arbitrarily selected the random forest to be used as the model in our application. This application is implemented in streamlit and contains a slightly modified version of this document as well as an interface to our selected model.</p>



