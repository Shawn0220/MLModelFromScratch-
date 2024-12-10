from starter import euclidean,cos_dis
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack


def construct_matrix(file_path='./movielens.txt', usecols=['user_id','movie_id','rating']):
    data = pd.read_csv(file_path, sep='\t', usecols=usecols)
    #extract all unqiue user_id & movie_id
    user_ids = data['user_id'].unique()
    all_movie_ids = range(1,1701)
    #mappings for user_id and movie_id:
    user_map = {user_id: i  for i,user_id in enumerate(user_ids)}
    movie_map = {movie_id: i for i, movie_id in enumerate(all_movie_ids)}
    #mapping 
    data['user_idx'] = data['user_id'].map(user_map)
    data['movie_idx'] = data['movie_id'].map(movie_map)
    #matrix construction
    rating_matrix = csr_matrix((data['rating'],(data['user_idx'],data['movie_idx'])),shape=(len(user_ids),1700))
    
    return rating_matrix, movie_map

def construct_user_demographic(user_map, file_path='./movielens.txt', usecols=['user_id', 'age', 'gender', 'occupation']):
    data = pd.read_csv(file_path, sep='\t', usecols=usecols)
    data['user_idx'] = data['user_id'].map(user_map)

    # Encode demographics (one-hot encoding for gender, occupation, normalized age)
    data = pd.get_dummies(data, columns=['gender', 'occupation'])
    data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()

    demographic_matrix = csr_matrix(data.drop(['user_id', 'user_idx'], axis=1).values)
    
    return demographic_matrix

def construct_movie_genre(movie_map, file_path='./movielens.txt', usecols=['movie_id', 'genre']):
    data = pd.read_csv(file_path, sep='\t', usecols=usecols)
    data['movie_idx'] = data['movie_id'].map(movie_map)

    # One-hot encode genres for each movie
    data = pd.get_dummies(data, columns=['genre']).groupby('movie_idx').sum()
    genre_matrix = csr_matrix(data.values)
    
    return genre_matrix

def construct_user_feature(movie_map, file_path='./train_c.txt', usecols=['user_id','movie_id','rating']):
    data = pd.read_csv(file_path, sep='\t', usecols=usecols)
    data['movie_idx'] = data['movie_id'].map(movie_map)
    
    user_ids = data['user_id'].unique()
    user_map = {user_id: i for i, user_id in enumerate(user_ids)}
    
    data['user_idx'] = data['user_id'].map(user_map)
    feature = csr_matrix((data['rating'],(data['user_idx'],data['movie_idx'])),shape=(len(user_ids),1700))
   
    return feature

def user_based_recommendation_optimized(ratings_matrix, user_feature, demographic_matrix, genre_matrix, alpha=0.5, beta=0.3, k=10, m=10):
    extended_ratings = vstack([ratings_matrix, user_feature])
    rating_similarity = cosine_similarity(extended_ratings)

    extended_demographics = vstack([demographic_matrix, csr_matrix(user_feature.shape)])
    demographic_similarity = cosine_similarity(extended_demographics)

    extended_genres = vstack([genre_matrix, csr_matrix((1, genre_matrix.shape[1]))])
    genre_similarity = cosine_similarity(extended_genres.T)  # Transpose for movie-based similarity

    new_user_idx = extended_ratings.shape[0] - 1

    # Combined similarity
    user_similarity = alpha * rating_similarity + beta * demographic_similarity + (1 - alpha - beta) * genre_similarity

    # Sorting similar users
    similar_users = np.argsort(-user_similarity[new_user_idx])[:k + 1]
    similar_users = similar_users[similar_users != new_user_idx]

    # Finding unseen movies
    new_user_array = user_feature.toarray().flatten()
    unrated_items = np.where(new_user_array == 0)[0]

    # Recommendation
    item_scores = {}
    for item in unrated_items:
        score_sum = 0
        sim_sum = 0
        for similar_user in similar_users:
            rating = ratings_matrix[similar_user, item]
            if rating > 0:
                score_sum += user_similarity[new_user_idx, similar_user] * rating
                sim_sum += user_similarity[new_user_idx, similar_user]
        
        if sim_sum > 0:
            item_scores[item] = score_sum / sim_sum

    recommended_items = sorted(item_scores, key=item_scores.get, reverse=True)[:m]

    return recommended_items

def user_based_recommendation(ratings_matrix , user_feature, k=10, m=10):
    extended_ratings = vstack([ratings_matrix, user_feature])
    user_similarity = cosine_similarity(extended_ratings)

    new_user_idx = extended_ratings.shape[0] - 1
    #sorting 
    similar_users = np.argsort(-user_similarity[new_user_idx])[:k+1] 
    similar_users = similar_users[similar_users != new_user_idx]  
    
    #figuring out unseen movie for new-user
    new_user_array = user_feature.toarray().flatten()  
    unrated_items = np.where(new_user_array == 0)[0] 
    
    #recommendation
    item_scores = {}
    for item in unrated_items:
        score_sum = 0
        sim_sum = 0
        for similar_user in similar_users:
            rating = ratings_matrix[similar_user, item]
            if rating > 0:
                score_sum += user_similarity[new_user_idx, similar_user] * rating
                sim_sum += user_similarity[new_user_idx, similar_user]
        
        if sim_sum > 0:
            item_scores[item] = score_sum / sim_sum 
    
    # m recommendations
    recommended_items = sorted(item_scores, key=item_scores.get, reverse=True)[:m]
    
    return recommended_items

def evaluate_recommendation(recommended_movies, validation_data, movie_map, threshold = 3):
    liked_items = validation_data[validation_data['rating'] >= threshold]['movie_id'].tolist()
    liked_items = [movie_map[key] for key in liked_items]
    print(recommended_movies)
    print(liked_items)
    true_positives = len(set(recommended_movies).intersection(set(liked_items)))
    precision = true_positives / len(recommended_movies)
    recall = true_positives / len(liked_items)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

if __name__ == '__main__':
    '''#construct global matrix
    matrix, movie_map = construct_matrix()
    #feature for user a
    feature_a = construct_user_feature(movie_map)

    #print(feature_a)
    #print(feature_a.shape)
    #recommended_item
    recommendation = user_based_recommendation(matrix, feature_a)
    validations = pd.read_csv('./valid_c.txt', sep='\t', usecols=['user_id','movie_id','rating'])
    precision, recall, f1_score = evaluate_recommendation(recommendation, validations, movie_map)
    print(f'precision:{precision}, recall:{recall}, f1_score:{f1_score}')'''
