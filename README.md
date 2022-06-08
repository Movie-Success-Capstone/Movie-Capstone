# Movie Ratings Capstone Project

##### Navigation: 
    
To view more details on any specific section, please click the 'Expand' drop-down arrow.
    
<details>
 
<summary>Table of Contents</summary>
<ul>
  <li><a href = '#intro'>Introduction</a></li>
  <li><a href = '#dict'>Data Dictionary </a></li>
  <li><a href = '#plan'>Planning </a></li>
  <li><a href = '#acquire'>Acquisition </a></li>
  <li><a href = '#prep'>Preparation </a></li>
  <li><a href = '#explore'>Exploration </a></li>
  <li><a href = '#model'>Modeling </a></li>
  <li><a href = '#conclusion'>Conclusion and Summary </a></li>
  <li><a href = '#steps'>Steps to Reproduce </a></li>
  <li><a href = '#appnx'>Appendix </a></li>
</ul>
</details>


<!-- <div id = 'intro'> -->
## Introduction 

<details>
<summary>Expand</summary>  
    
### Classification Model for Predicting Movie Success 
    
After the recent pandemic the movie industry has been slow to return to the volume that it once produced at.  Using open source movie databases, we set out to determine a movie's success based on its financial performance as well as its average viewer ratings.   Keeping an eye for any trends from previous successful movies, we then constructed a Machine Learning(ML) Model to predict the success of other movies.  
All of this can be used to help determine how to better invest in projects going forwards and what parameters can be set to lead to a healthy return on investments. 

    
</details>
<!-- </div> -->
<!-- End Introduction here  -->
<hr>

<!-- <div id = 'dict'> -->
## Data Dictionary

<details>
<summary>Expand</summary>
      
#### Original Dataset
    
| Feature  | Description | Data Type | 
| :------------- | :------------- | :------------- |
| Title  | Movie title  | Object  |
| Success_rating  | Scaled parameter iMDb uses to evaluate movie success  | Float64  |
| Genres  | Movie classification type  | Object  |
| Budget  | Amount in U.S. dollar spend in the production of the movie  | Float64  |
| Revenue  | The total U.S. dollar amount collected after a movie release  | Float64  |
| Vote_average  | The average voting of a movie  | Float64  |
| Vote_count  | The total vount count of a movie  | Float64  |
| Production_companies  | Name(s) of production company tasked with creation of movie  | Object  |
| Production_countries  | Country a movie was marketed/ played   | Object  |
| Overview  | The overview description of a movie  | Object  |
| Popularity  | Scaled numerical measure of perceived movie likability  | Float64  |
| Runtime  | Recorded movie play-time. (How long the movie is)  | Float64  |
| Release_date  | Specific calendar date a movie was released. (YYYY-MM-DD)  | Object**  |

    
#### Engineered Features
    
| Feature  | Description | Data Type | 
| :------------- | :------------- | :------------- |
| Success  | TARGET VARIABLE == (Revenue / (Budget * 2)) * Vote_Average  | Bool  |
| Profit_amount  | U.S. dollar amount calculated from subtracting budget from revenue  | Float64  |
| Profitable  | Revenue - budget > than 0 means profitable  | Bool  |
| Cast_actor_1  | cast actor list 1  | Object  |
| Cast_actor_2  | cast actor list 2  | Object  |    
| Cast_actor_3  | cast actor list 3  | Object  |
| Total_n_cast  | cast actor list 4  | Float64  |
| Release_year  | The year a specific movie was released for general public consumption/ enjoyment  | Int64**  |
| Month  | Month of the year a movie was released to general public  | int64**  |
| Runtime.1  | The runtime of a movie. | Float64  |


    ** indicates datatype maybe converted to datetime format
</details>
<!-- </div>  -->
<!-- End Data Dictionary here  -->

<hr>

<!-- <div id = 'plan'> -->
## Planning Phase
<!-- <details>
<summary>Expand</summary> -->
    

    
#### (a). Project Goal
   This project aims to achieve the main goal of modeling the prediction of successful movies by applications of scientific, statistical and adaptations of business logic in formation of this final model. A thorough thoughout planning phase involved several main considerations as follows:
    
#### (b). Project Description
   After the recent pandemic the movie industry has been slow to return to the volume that it once produced at. Using open source movie databases, we set out to determine a movie's success based on its financial performance as well as its average viewer ratings. Keeping an eye for any trends from previous successful movies, we then constructed a Machine Learning(ML) Model to predict the success of other movies. All of this can be used to help determine how to better invest in projects going forwards and what parameters can be set to lead to a healthy return on investments. 
    
#### (c). Methodology
    
   Plan >> Acquire >> Prepare >> Explore >> Model >> Deliver
    
#### (d). Exploration Questions of interest
    
> - Is there a relationship between budget and revenue?
> - How runtime affect movie success?
> - What are the top 5 Highest Voted Movies?
> - What are top 5 Highest profit movies?
> - Which genre has the highest profit? Or top 5?

#### (e). Target Variable 
       TARGET VARIABLE == (Revenue / (Budget * 2)) * Vote_Average
       
       Target column == success column

The matrix that we will be using to evaluate our model contains two perspectives. First, we must make sure that the movie is a financial success.  It is not uncommon for most popular movies today to see a 100% return on investment, with revenue reaching twice of what was put into the final product. So we set this as the standard to base our expectations on. 

Secondly, we must consider a movie’s overall impression on the general public.  Film popularity is the most obvious manifestation of audience taste, and it is based upon the 'willingness-to-pay’. For this purpose we set the average rating (based on a 1-10 scale) as a multiplier for the movies financial success.  Multiplying by the rating will help any films that performed poorly in the theatres, but have gained popularity over time therefore increase the overall value of the film itself. (ex. Cult Classics)

This formula will help us rank movie’s success from negative to positive, the higher score the more successful a movie is. We decided to use score 6.5 to evaluate a movie’s success. There are about 38% of movies in our dataset that meet this criteria. This will encapsulate all fiscally successful projects, not just the critically acclaimed films at the very top.  Because the truth is that there are thousands off projects at all levels of media, and they all have the potential to generate value for those that make them. We want to capture what can help keep you in that 38% of film population.  

   
#### (f). Stakeholders
> - Movie producers and interested general public
    
<!-- </details>    -->
<!-- </div> -->
<!-- End Planning here  -->

<hr>

<!-- <div id = 'acquire'> -->
## Data Acquisition
<details>
<summary>Expand</summary>
    
The data for this project was acquired from open-source Kaggle website- https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset. This set consisted of more than 5000 data points with 28 attributes. At this time, no Application Programming Interface (API) is utelized in streamlined acquisition process due to required iMDb policies, however, in the future project updates we intend to implements APIs in simplicity of rep-producing this preject. With this stated, directly download and save locally in the same project folder the following comma-sepated files(csv):
    
- Credits.csv 
- Movies_metadata.csv
- Keywords.csv
- Ratings.csv
    
In the prepare phase in this README.md file, we will describe the joining procedure followed in the joining of these separate csv files into  of the final dataframe.
</details>
</div>
<!-- End Acquire here  -->

<hr>

<!-- <div id = 'prep'> -->
## Data Praparation 
<details>
<summary>Expand</summary>
    
The parent module for both data acquisition and preparation are included in the final)acquire module. Within the same module file, specific tasts are divided by individualized function to better enhance readability. __Wrangle_df__ function is the resultant that collectively hosts calls to the main __prep_data__ function function for the our data preparation. This function uses local data caching method to enhance data loading speeds. 

#### Prep_data function

- Drops unnecessary columns 
- Drops individual row nulls and any duplicated values 
- Applies median budget values for budget between 0 to 1,000,000
- Appends names(with whitespace) on genres columns
- Returns profitable as type bool for explorations
- Extracts nested dictionary data from columns production_company and cast
- One hot ecode data for modeling
- Feature engineer columns:
    
   * Release_year
   * Release_date
   * Profitable   
   * Success 
   * Profit_amount {revenue - budget}
   * Success_rating {(revenue / budget) * 2] * vote_average}
    
- Sets dataframe index as __id__
- Saved a __clean.csv__ file for explorations. 
- Explain variables as defined in the project through graphical data dictionary representation
   
#### Train_validate_test_split function
- Splits the dataset into train, validate, and test sets for exploration and modeling.
    
    
</details>
<!-- </div> -->
<!-- End Prepare here  -->

<hr>

<!-- <div id = 'explore'> -->
## Exploration 
<details>
<summary>Expand</summary>

Reference to the project main goal of model prediction of movie success between 1915 t0 2017, this exploration phase was key in understanding factors that predict movie success. Guiding our predictions, the following questions were initially analyzed to determine pattern and relations among features of interest:
    
- Is there a relationship between budget and revenue?
- What are the top 5 Highest Voted Movies?
- Examining revenue, what are the top 5 highest revenue movies?
- Which top 5 movie genres that are likely to yield the highest profits?
    
#### Key Findings 
    
- Budget and revenue shows elevated corelations
- Vote count and also has hightened corellations with both revenue and profit amount
- Very little correlation between budget and success rating and also between vote average and budget
- Most voted movies are:
    
   * Minions
   * Wonder Woman
   * Beauty and the Beast
   * Baby Driver
   * Big Hero 6
    
- Top 5 most revenue generators movies are:
        
   * Avator
   * Star Wars: The Force Awakens
   * Titanic
   * The Avengers
   * Jurassic World


</details>
<!-- </div> -->
<!-- End Explore here  -->

<hr>

<!-- <div id = 'model'> -->
## Modeling 
<details>
<summary>Expand</summary>
    Exploration phase identified arrays of possible divers for a movie success. In this section, we will create a machine learning algorithm model that better predicts movie success and use our findings as recommendations for our stakeholders. Three supervised machine learning classifications models were created in this project:
* Logistic regression
* K-Nearest Neighbor (KNN)
* Decision tree
A baseline model for our project was created from engineered columns of success as a measure of overall performance of the models above. Our definition of a successful movie is guided by financial metrics a movie generates as captured by the dataset and 'perceived success' as expressed by features such as ratings, votes among othes. As is, the dataset baseline movie success accounts to 38.90%.
The following sections will tabulate the models results over the train and validat subset and later test the best model over test subset to limit data leakage.

</details>
<!-- </div> -->
<!-- End Modeling here  -->

<hr>

<!-- <div id = 'conclusion'> -->
## Conclusion and Recommendations 
<details>
<summary>Expand</summary>
    
</details>
<!-- </div> -->
<!--   -->
Best model is logistic regession based on precision metrics. Our formulated model beats the baseline of 62.09% by 15.27 percentile points. In summary, our team was successful in building a success-predicting ML Model with which  valuable insights could be derived. Our best model (Logistic Regression) predicted movie success with 77.36% precision, and 15% increase from the baseline performance.
We recommend to our stakeholders the following measurements, guidelines that can improve the chances of a filmmakers project overall success. 
Ideal Runtime —  180 - 230 minutes       
Ideal Total Cast Size —  130 - 180 people
Consistently Successful Genres — Action, Adventure, Comedy
Least Profitable/Successful — Documentary, Mystery
Future project versions intend to understand drivers for the low success in these least performing categories and offer guidance on measures to improve.  We also discerned the most common actors for successful movies, which with time could give details on what personalities draw in more revenue.
<hr>

<!-- <div id = 'steps'> -->
## Steps to Reproduce Project 
<details>
<summary>Expand</summary>
  <ol>
      <li>Step 1.  Clone this repository into your local machine using the following command:
git clone git@github.com:Movie-Success-Capstone/Movie-Capstone.git</li>
      <li>Step 2.You will need Pandas, Numpy, Matplotlib, Seaborn, and SKLearn installed on your machine. </li>
      <li>Step 3. Please run python acquire.py in a terminal to acquire the csv file.</li>
      <li>Step 4. Now you can start a Jupyter Notebook session and execute the code blocks in the final_report.ipynb notebook.
</li>
  </ol>
</details>
<!-- </div> -->
<!--  -->
  
<hr> 

<!-- <div id = 'appnx'> -->
## Appendix 
<!-- <details> -->
<summary>Links</summary>
    <a href = 'https://github.com/Movie-Success-Capstone/Movie-Capstone/blob/main/Final/final_report.ipynb'>Github</a>
    <br>
    <a href = 'https://www.canva.com/design/DAFCeimVECI/h4M50njErHgR5OBCSX1bZg/edit?utm_content=DAFCeimVECI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton'>Project Brochure</a>
    <br>
    <a href = 'https://www.canva.com/design/DAFC11J4Ygw/c4zUuaY4IVLyh8VSIOm7gw/edit?utm_source=shareButton&utm_medium=email&utm_campaign=designshare'>Presentation Slides</a>
<!-- </details> -->
<!-- </div> -->
<!-- End Appendix here  -->

<div id ='top'></div>
