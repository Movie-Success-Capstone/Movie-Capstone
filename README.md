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
    
### Cinematic Success: A Matter of Classification
    
Cinema is a captivating art form. The traditions, techniques and practices which envelop it vary across cultures. Some films garner international acclaim, while others are widely discredited as failures. So this begs the question, are there seemingly universal features which help to determine any given film's success--whether in terms of income amount generated via film sales (i.e., revenue), consumer feedback, or some other feature?
    Surely, springs of ideas would come to mind for anyone pondering this with their eyes peeled: that's attentive viewership in a nutshell. So stick around until we roll the credits, and as the exit music plays we will give you some insights toward predicting if a movie would move money into its makers' pockets and joy into its viewers' hearts.  
    [END SCENE]
                              CUT TO: 
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
| Vote_average  | ..........  | Float64  |
| Vote_count  | ...........  | Float64  |
| Production_companies  | Name(s) of production company tasked with creation of movie  | Object  |
| Production_countries  | Country a movie was marketed/ played   | Object  |
| Overview  | ...........  | Object  |
| Popularity  | Scaled numerical measure of perceived movie likability  | Float64  |
| Runtime  | Recorded movie play-time. (How long the movie is)  | Float64  |
| Release_date  | Specific calendar date a movie was released. (YYYY-MM-DD)  | Object  |

    
#### Engineered Features
    
| Feature  | Description | Data Type | 
| :------------- | :------------- | :------------- |
| Success  | ...........  | Bool  |
| Profit_amount  | U.S. dollar amount calculated from subtracting budget from revenue  | Float64  |
| Profitable  | ..........  | Bool  |
| Cast_actor_1  | ............  | Object  |
| Cast_actor_2  | ............  | Object  |    
| Cast_actor_3  | ............  | Object  |
| Total_n_cast  | ..............  | Float64  |
| Release_year  | The year a specific movie was released for general public consumption/ enjoyment  | Int64**  |
| Month  | Month of the year a movie was released to general public  | int64**  |
| Runtime.1  | ..........  | Float64  |

    
</details>
<!-- </div>  -->
<!-- End Data Dictionary here  -->

<hr>

<!-- <div id = 'plan'> -->
## Planning Phase
<!-- <details>
<summary>Expand</summary> -->
    
This project aims to achieve the main goal of modeling the prediction of successful movies by applications of scientific, statistical and adaptations of business logic in formation of this final model. A thorough thoughout planning phase involved several main considerations as follows:
    
#### (a). Project Goal
    Text goes here...    
    
#### (b). Project Description
    Text goes here...    
    
#### (c). Methodology
    
    Plan >> Acquire >> Prepare >> Explore >> Model >> Deliver
    
#### (d). Exploration Questions of interest
    
> - Is there a relationship between budget and revenue?
> - How runtime affect movie success?
> - What are the top 5 Highest Voted Movies?
> - What are top 5 Highest profit movies?
> - Which genre has the highest profit? Or top 5?
> - How does production company affect profit?
    
#### (e). Target Variable 
    Text goes here
   
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
    
> - Credits.csv 
> - Movies_metadata.csv
> - Keywords.csv
> - Ratings.csv
    
In the prepare phase in this README.md file, we will describe the joining procedure followed in the joining of these separate csv files into  of the final dataframe.
</details>
</div>
<!-- End Acquire here  -->

<hr>

<!-- <div id = 'prep'> -->
## Data Praparation 
<details>
<summary>Expand</summary>
    
Data wrangling (preparation) for the iMDb involved several steps 

</details>
<!-- </div> -->
<!-- End Prepare here  -->

<hr>

<!-- <div id = 'explore'> -->
## Exploration And Feature Engineering 
<details>
<summary>Expand</summary>
    Explore goes here...
</details>
<!-- </div> -->
<!-- End Explore here  -->

<hr>

<!-- <div id = 'model'> -->
## Modeling 
<details>
<summary>Expand</summary>
    Modeling goes here...
</details>
<!-- </div> -->
<!-- End Modeling here  -->

<hr>

<!-- <div id = 'conclusion'> -->
## Conclusion And Recommendations 
<details>
<summary>Expand</summary>
    Conclusion goes here...
</details>
<!-- </div> -->
<!-- End Conclusion here  -->

<hr>

<!-- <div id = 'steps'> -->
## Steps to Reproduce Project 
<details>
<summary>Expand</summary>
  <ol>
      <li>Step 1</li>
      <li>Step 2</li>
      <li>Step 3</li>
  </ol>
</details>
<!-- </div> -->
<!-- End Steps here  -->
  
<hr> 

<!-- <div id = 'appnx'> -->
## Appendix 
<details>
<summary>Links</summary>
    <a href = 'https://github.com/Movie-Success-Capstone/Movie-Capstone'>Github</a>
    <br>
    <a href = 'https://github.com/Movie-Success-Capstone/Movie-Capstone'>Google Slides</a>
</details>
<!-- </div> -->
<!-- End Appendix here  -->

<div id ='top'></div>
