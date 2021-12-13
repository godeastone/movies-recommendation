import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import pylab as pl
from mlxtend.frequent_patterns import apriori, association_rules
from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings; warnings.simplefilter('ignore')


"""
Function : ErrorLog

Parameter:
    string
        error : error message

Return:
"""
def ErrorLog(error):
    current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
    with open("Log.txt", "a") as f:
        f.write(f"[{current_time}] - {error}\n")


"""
Function : find_movieId

Parameter:
    Data Frame
        _input_movies : entered movie list by users
        _movie_df : metadata of movies

Return:
        return list of movie ids
"""
def find_movieId(_input_movies, _movies_df):
    indices =[]

    for movie in _input_movies:
        index = _movies_df.loc[_movies_df["original_title"] == movie, 'id'].values[0]
        indices.append(index)

    return indices


"""
Function : drop_trash_data

Parameter:
    Data Frame
        _movie_df : metadata of movies

Return:
        return trash removed dataframe
"""
def drop_trash_data(_movie_df):
    _movie_df.drop(_movie_df.index[19730], inplace=True)
    _movie_df.drop(_movie_df.index[29502], inplace=True)
    _movie_df.drop(_movie_df.index[35585], inplace=True)

    return _movie_df


"""
Function : apriori_encoding

Parameter:
    int64
        r : rating of movie

Return:
        return 0 or 1
"""
def apriori_encoding(r):
    if r <= 0:
        return 0
    elif r >= 1:
        return 1


"""
Function : do_apriori

Parameter:
        _input_movies : entered movie list by users
        _movies_df : movies_metadata.csv file
        _ratings_df : ratings_small.csv file

Return:
        return _apriori_result : result of A-priori algorithm
"""
def do_apriori(_input_movies, _movies_df, _ratings_df):
    # Internal variables
    _apriori_result = []

    """ Remove the Nan title & join the dataset """
    Nan_title = _movies_df['title'].isna()
    _movies_df = _movies_df.loc[Nan_title == False]

    _movies_df = _movies_df.astype({'id' : 'int64'})
    df = pd.merge(_ratings_df, _movies_df[['id', 'title']], left_on='movieId', right_on='id')
    df.drop(['timestamp', 'id'], axis=1, inplace=True)

    """ Prepare Apriori
        row : userId | col : movies """
    df = df.drop_duplicates(['userId', 'title'])
    df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0)
    df_pivot = df_pivot.astype('int64')
    df_pivot = df_pivot.applymap(apriori_encoding)
    # print(df_pivot.head())

    """ A-priori Algorithm """
    #calculate support and eradicate under min_support
    frequent_items = apriori(df_pivot, min_support=0.07, use_colnames=True)
    # print(frequent_items.head())

    # using association rules, compute the other parameter ex) confidence, lift ..
    association_indicator = association_rules(frequent_items, metric="lift", min_threshold=1)

    # sort by order of lift
    df_lift = association_indicator.sort_values(by=['lift'], ascending=False)
    # print(df_res.head())

    """ Start recommendation """
    for selected_movie in _input_movies:
        num = 0
        df_selected = df_lift[df_lift['antecedents'].apply(lambda x: len(x) == 1 and next(iter(x)) == selected_movie)]
        df_selected = df_selected[df_selected['lift'] > 1.2]
        recommended_movies = df_selected['consequents'].values

        for movie in recommended_movies:
            for title in movie:
                if title not in _apriori_result and num < 10:
                    _apriori_result.append(title)
                    num += 1

    return _apriori_result


"""
Function : do_kmeans

Parameter:
        _apriori_result : candidate movies; result of apriori algorithm
        _input_movies : entered movie list by users
        _movies_df : movies_metadata.csv file

Return:
        return _kmeans_result All movie's cluster indexes in _input_movies
"""
def do_kmeans(_apriori_result, _input_movies, _movies_df):
    # record all clusters in _input_movies
    clusters = []
    _kmeans_result = []

    numeric_df = _movies_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'title']]

    numeric_df.isnull().sum()
    numeric_df.dropna(inplace=True)
    # print(df_numeric['vote_count'].describe())

    """cut off the movies' votes less than 25"""
    df_numeric = numeric_df[numeric_df['vote_count'] > 25]

    # Normalize data - by MinMax scaling provided by sklearn
    minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title', axis=1))
    df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])

    """Apply K-means clustering"""
    # make elbow curve to determine value 'k'
    num_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in num_cluster]
    score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]

    # print elbow curve
    pl.plot(num_cluster, score)
    pl.xlabel("Number of clusters")
    pl.ylabel("Score")
    pl.title("Elbow curve")
    #plt.show()  # maybe k=4 is appropriate

    # Fit K-means clustering for k=5
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_numeric_scaled)  # result is kmeans_label

    # write back labels to the original numeric data frame
    df_numeric['cluster'] = kmeans.labels_
    # print(df_numeric.head())

    # Search all clusters in user selected movies
    for movie1 in _input_movies:
        try:
            cluster_candid = df_numeric.loc[df_numeric["title"] == movie1, 'cluster'].values[0]
            # print(cluster_candid)
            clusters.append(cluster_candid)
        except IndexError as e:
            msg = "There is No cluster in movie [" + movie1 + ']'
            ErrorLog(msg)
            #print(msg)

    # Filtering movies that are not in clusters
    for movie2 in _apriori_result:
        try:
            cluster_tmp = df_numeric.loc[df_numeric["title"] == movie2, 'cluster'].values[0]
            if cluster_tmp in clusters:
                _kmeans_result.append(movie2)
        except IndexError as e:
            msg = "There is No cluster in movie [" + movie2 + ']'
            ErrorLog(msg)
            #print(msg)

    return _kmeans_result


"""
Function : compute_CF

Parameter:
        movie_name : selected movie in input movies
        _movies_users_compressed : pairs of ratings that computed by csr_matrix
        _model : KNN models with cosine metric
        _n_recommendations : number of recommendations
        _movie_names : all movie names in ratings_small.csv

Return:
        return [_n_recommendations] number of recommended movies by models
"""
def compute_CF(_movie, _movies_users_compressed, _model, _n_recommendations, _movie_names):
    recommend_frame = []

    _model.fit(_movies_users_compressed)
    movie_index = process.extractOne(_movie, _movie_names['title'])[2]

    try:
        # find the closest movie compare to selected movie
        distances, indices = _model.kneighbors(_movies_users_compressed[movie_index], n_neighbors=_n_recommendations)
    except IndexError as e:
        msg = "There is No movie [" + _movie + '] in csr_matrix'
        #print(msg)
        ErrorLog(msg)
        return []

    rec_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                key=lambda x: x[1])[:0:-1]

    for index in rec_indices:
        recommend_frame.append({'Title': _movie_names['title'][index[0]], 'Distance': index[1]})

    df = pd.DataFrame(recommend_frame, index=range(1, _n_recommendations))
    result = df['Title'].tolist()

    return result


"""
Function : do_collaborative_filtering

Parameter:
        _input_movies : entered movie list by users
        _ratings_df : ratings_small.csv file
        _movies_df : movies_metadata.csv file

Return:
        return recommended movies by collaborative filtering
"""
def do_collaborative_filtering(_input_movies, _ratings_df, _movies_df):
    _collab_result = []
    ratings_data = _ratings_df.drop('timestamp', axis=1)
    movie_names = _movies_df[['title', 'genres']]

    movies_users = ratings_data.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)

    # by using csr_matrix, compress the sparse data frame
    movies_users_compressed = csr_matrix(movies_users.values)
    #print(movies_users_compressed)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

    # make a similar movie groups based on ratings
    model.fit(movies_users_compressed)

    for movie in _input_movies:
        num_recom = 4
        recommend = compute_CF(movie, movies_users_compressed, model, num_recom, movie_names)
        _collab_result.extend(recommend)

    return _collab_result


"""
Function : main

Parameter: _

Return:
        return : recommended movies
"""
def main(input_movies):
    final_result = ""
    """
    # Make user selected movies list by parsing input arguments
    for input_movie in sys.argv[1:]:
        input_movies.append(input_movie)
    """
    final_result += "Selected movies (5 movies) : " + ",".join(input_movies) + "\n\n"
    print(final_result)

    # Read csv files
    movies_df = pd.read_csv('data/movies_metadata.csv')
    ratings_df = pd.read_csv('data/ratings_small.csv')

    # Drop the trash(error) data
    movies_df = drop_trash_data(movies_df)

    # recommend based on a-priori & k-means
    apriori_result = do_apriori(input_movies, movies_df, ratings_df)
    kmeans_result = do_kmeans(apriori_result, input_movies, movies_df)

    # recommend based on collaborative filtering
    collabo_result = do_collaborative_filtering(input_movies, ratings_df, movies_df)

    final_result += "A-priori & K-means clustering recommend movie : " + ",".join(kmeans_result) + "\n\n"
    final_result += "Collaborative filtering recommend movie : " + ",".join(collabo_result) + "\n\n"

    print(final_result)
    f = open("result.txt", "w")
    f.write(final_result)
    f.close()

    return final_result




if __name__ == '__main__':
    main()