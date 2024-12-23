# LetterboxdData-DSA210Project

## Project Overview

This project aims to analyze my personal **Letterboxd ratings** data to uncover patterns in my movie preferences. Specifically, I will explore how I rate movies based on **genre** and **release year** (e.g., by decade). By visualizing the data and performing statistical analysis, I will identify which genres and years of movies are most appealing to me, as well as how my ratings vary across different categories.

The analysis will focus on the **5-star rating system** used on Letterboxd, and I will explore whether my preferences have changed over time or if certain genres tend to receive higher ratings. This project will also include hypothesis testing to assess whether my ratings differ between movies released before and after that year (I didn't choose yet).

## Dataset

The dataset for this project will be my **Letterboxd film ratings**, which can be accessed through my [Letterboxd profile](https://letterboxd.com/eminthesecond/films/). The data includes the following attributes for each movie:

- **Title**: The name of the movie
- **Genre(s)**: One or more genres assigned to the movie
- **Release Year**: The year the movie was released
- **Rating**: The rating I assigned to the movie (on a 5-star scale)

I will extract the data from my Letterboxd profile using either the **Letterboxd API** or by exporting my ratings in CSV format (whichever method is most efficient). Once the data is collected, I will clean and preprocess it to ensure consistency, including handling missing data and standardizing genre labels.

### Data Considerations:
- The dataset includes movies rated by me, but it may exclude entries categorized as TV series or short films by me.
- I plan to analyze a subset of my most highly rated movies, but will aim to include a broad enough sample to detect meaningful trends.
  
## Tools and Technologies

- **Python** for data analysis and visualization
- **Pandas** for data manipulation and cleaning
- **Matplotlib** and **Seaborn** for creating visualizations (e.g., plots of average ratings by genre or release year)
- **SciPy** for hypothesis testing (e.g., t-tests to compare ratings before and after that year)

## Analysis Plan

1. **Data Collection**:
   - I will extract my Letterboxd ratings using either the **Letterboxd API** or a CSV export. I will then load the data into a Pandas DataFrame and clean it (e.g., handling missing values, normalizing genres).

2. **Visualization**:
   - Create plots to visualize trends in my movie ratings by **release year** and **genre**. Examples of visualizations include:
     - A time series plot showing median ratings over the years
     - A bar plot showing average ratings by genre
     - A scatter plot comparing ratings with release year for different genres

3. **Hypothesis Testing**:
   - I will test the following hypothesis:
     - **H₀**: My average ratings for movies before and after that year are the same.
     - **Hₐ**: Movies released after that year a receive higher ratings on average than movies released before that year.
   - This will be tested using a two-sample t-test to compare the means of the two groups (pre-that year vs. post-that year movies).

4. **Genre Analysis**:
   - I will analyze which **genres** consistently receive the highest ratings. This will involve calculating the average or median ratings for each genre and identifying patterns, such as whether I rate certain genres (e.g., drama, sci-fi, comedy) higher than others.
   - I will also investigate whether my preference for specific genres has changed over time.

5. **Additional Explorations**:
   - If time allows, I may explore additional factors such as:
     - **Directors and Actors**: Are there certain directors or actors whose films consistently receive higher ratings?
     - **Movie Length**: Do I tend to rate shorter or longer films higher?
   
## Example Plot

Below is an example of a plot I will generate to visualize the relationship between **movie release years** and my **average ratings**:
