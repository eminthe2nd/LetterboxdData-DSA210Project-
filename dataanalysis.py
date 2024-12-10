import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('eminthesecond-films-by-date.csv')

data['Genres'] = data['Genres'].fillna('[]')
data['Genres'] = data['Genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)

unique_genres = set(genre for sublist in data['Genres'] for genre in sublist)

watches_index = data.columns.get_loc('Watches')

data = data.drop(columns=data.columns[watches_index + 1:])

genre_mapping = {
    'Action-packed space and alien sagas': 'Action',
    'Bloody vampire horror': 'Horror',
    'Dangerous technology and the apocalypse': 'Science Fiction',
    'Gory, gruesome, and slasher horror': 'Horror',
    'Horror, the undead and monster classics': 'Horror',
    'Monsters, aliens, sci-fi and the apocalypse': 'Science Fiction',
    'Sci-fi horror, creatures, and aliens': 'Science Fiction',
    'Superheroes in action-packed battles with villains': 'Action',
    'Survival horror and zombie carnage': 'Horror',
    'Twisted dark psychological thriller': 'Thriller',
    'Show All…': None
}

def normalize_genres(genres):
    normalized = []
    for genre in genres:
        if genre in genre_mapping:
            mapped_genre = genre_mapping[genre]
            if mapped_genre:  
                normalized.append(mapped_genre)
        else:
            normalized.append(genre)
    return list(set(normalized))  

data['Normalized_Genres'] = data['Genres'].apply(normalize_genres)

data[['Genres', 'Normalized_Genres']].head()

data['Average_rating'] = data['Average_rating'].fillna(data['Average_rating'].median())
data['Runtime'] = data['Runtime'].fillna(data['Runtime'].median())

data['Decade'] = (data['Release_year'] // 10) * 10

data.isnull().sum()

genre_avg_ratings = data.explode('Normalized_Genres').groupby('Normalized_Genres')['Average_rating'].mean().sort_values()
genre_avg_ratings.plot(kind='barh', figsize=(10, 6), title='Average Ratings by Genre')
plt.xlabel('Average Rating')
plt.show()

decade_avg_ratings = data.groupby('Decade')['Average_rating'].mean()
decade_avg_ratings.plot(kind='line', figsize=(10, 6), marker='o', title='Average Ratings by Decade')
plt.ylabel('Average Rating')
plt.xlabel('Decade')
plt.show()

genre_counts = data.explode('Normalized_Genres')['Normalized_Genres'].value_counts()
genre_counts.plot(kind='barh', figsize=(10, 6), title='Number of Ratings by Genre')
plt.xlabel('Number of Ratings')
plt.show()

data.explode('Normalized_Genres').groupby('Normalized_Genres')['Average_rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)

sns.scatterplot(data=data, x='Runtime', y='Average_rating')
plt.title('Runtime vs. Average Rating')
plt.show()

yearly_avg_ratings = data.groupby('Release_year')['Average_rating'].mean()
yearly_avg_ratings.plot(figsize=(12, 6), marker='o', title='Yearly Trend of Ratings')
plt.ylabel('Average Rating')
plt.xlabel('Year')
plt.show()

year_split = 2000
pre_year = data[data['Release_year'] < year_split]['Average_rating']
post_year = data[data['Release_year'] >= year_split]['Average_rating']

t_stat, p_value = ttest_ind(pre_year, post_year, nan_policy='omit')
print(f"T-statistic: {t_stat}, P-value: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis: Ratings differ before and after the year.")
else:
    print("Fail to reject the null hypothesis: No significant difference in ratings.")

pre_year_avg = pre_year.mean()
post_year_avg = post_year.mean()

print(f"Average Rating Before {year_split}: {pre_year_avg}")
print(f"Average Rating After {year_split}: {post_year_avg}")

labels = ['Before', 'After']
averages = [pre_year_avg, post_year_avg]

plt.bar(labels, averages, color=['green', 'blue'])
plt.title(f'Average Ratings Before and After {year_split}')
plt.ylabel('Average Rating')
plt.show()

genre_decade_trends = data.explode('Normalized_Genres').groupby(['Decade', 'Normalized_Genres'])['Average_rating'].mean().unstack()
genre_decade_trends.plot(figsize=(12, 8), title='Average Ratings by Genre Over Decades')
plt.ylabel('Average Rating')
plt.show()

bins = [0, 60, 90, 120, 150, 180, float('inf')]
labels = ['<1 hr', '1-1.5 hrs', '1.5-2 hrs', '2-2.5 hrs', '2.5-3 hrs', '>3 hrs']
data['Runtime_bin'] = pd.cut(data['Runtime'], bins=bins, labels=labels)

runtime_avg_ratings = data.groupby('Runtime_bin')['Average_rating'].mean()
runtime_avg_ratings.plot(kind='bar', figsize=(10, 6), color='purple', title='Average Ratings by Runtime')
plt.ylabel('Average Rating')
plt.show()

director_avg_ratings = data.groupby('Director')['Average_rating'].mean().sort_values(ascending=False).head(10)
print(director_avg_ratings)
director_avg_ratings.plot(kind='barh', figsize=(10, 6), title='Top 10 Directors by Average Rating')
plt.xlabel('Average Rating')
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Runtime', 'Average_rating']])

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Runtime', y='Average_rating', hue='Cluster', data=data, palette='Set1')
plt.title('Clustering of Movies Based on Runtime and Rating')
plt.show()

X = pd.get_dummies(data[['Runtime', 'Release_year', 'Decade']])
y = data['Average_rating']

model = LinearRegression()
model.fit(X, y)

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

importances = pd.DataFrame(rf_model.feature_importances_, X.columns, columns=['Importance'])
importances.sort_values(by='Importance', ascending=False).head(10)

predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

sns.pairplot(data[['Average_rating', 'Runtime', 'Release_year']])
plt.show()

exploded_data = data.explode('Normalized_Genres')

plt.figure(figsize=(12, 6))
sns.boxplot(data=exploded_data, x='Normalized_Genres', y='Average_rating')
plt.title('Ratings Distribution by Genre')
plt.xticks(rotation=90)
plt.show()


yearly_avg_ratings = data.groupby('Release_year')['Average_rating'].mean()
plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_ratings)
plt.title('Average Ratings by Year')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

genre_decade_trends = data.explode('Normalized_Genres').groupby(['Decade', 'Normalized_Genres'])['Average_rating'].mean().unstack()
plt.figure(figsize=(12, 8))
sns.heatmap(genre_decade_trends, cmap='YlGnBu', annot=True)
plt.title('Genre Preferences Over Time')
plt.xlabel('Genre')
plt.ylabel('Decade')
plt.show()

data.info()
data.describe()

missing_data = data.isnull().sum()
print(missing_data)

sns.histplot(data['Average_rating'], kde=True, bins=10)
plt.title('Distribution of Ratings')
plt.show()

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

mean_diff = post_year.mean() - pre_year.mean()
pooled_std = np.sqrt((pre_year.var() + post_year.var()) / 2)
effect_size = mean_diff / pooled_std
print(f"Cohen's d (Effect Size): {effect_size}")

data['Is_Post_Year'] = (data['Release_year'] >= year_split).astype(int)
X = data[['Runtime', 'Is_Post_Year']]
X = sm.add_constant(X)
y = data['Average_rating']

model = sm.OLS(y, X).fit()
print(model.summary())

data.info(), data.head()

data.to_csv('cleaned_letterboxd_data.csv', index=False)






