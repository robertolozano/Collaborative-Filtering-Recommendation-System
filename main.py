import numpy as np
from collections import defaultdict

### README: TO USE PROJECT SIMPLY PLACE TEST AND TRAIN FILES IN SAME FOLDER AS THIS FILE

def load_training_data(file_path):
    user_ratings = defaultdict(dict)
    movie_count = defaultdict(int)
    movie_ratings = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            user, movie, rating = map(int, line.strip().split())
            user_ratings[user][movie] = rating
            movie_count[movie] += 1
            movie_ratings[movie].append(rating)
    return user_ratings, movie_count, movie_ratings

def load_test_data(file_path):
    test_data = []
    with open(file_path, 'r') as file:
        for line in file:
            user, movie, rating = map(int, line.strip().split())
            test_data.append((user, movie, rating))
    return test_data

def apply_iuf(ratings, movie_count, total_users):
    iuf_ratings = {}
    for movie, rating in ratings.items():
        iuf_factor = np.log(total_users / (1 + movie_count[movie]))
        iuf_ratings[movie] = rating * iuf_factor
    return iuf_ratings

def apply_variance_adjustment(ratings, movie_variance):
    var_adjusted_ratings = {}
    for movie, rating in ratings.items():
        if movie in movie_variance:
            var_adjustment_factor = np.sqrt(movie_variance[movie])
            var_adjusted_ratings[movie] = rating * var_adjustment_factor
        else:
            var_adjusted_ratings[movie] = rating  
    return var_adjusted_ratings


def calculate_variance(movie_ratings):
    movie_variance = {}
    for movie, ratings in movie_ratings.items():
        movie_variance[movie] = np.var(ratings)
    return movie_variance

def cosine_similarity(user1, user2):
    common_movies = set(user1.keys()).intersection(set(user2.keys()))
    if not common_movies:
        return 0

    dot_product = sum(user1[movie] * user2[movie] for movie in common_movies)
    norm_user1 = sum(user1[movie] ** 2 for movie in common_movies) ** 0.5
    norm_user2 = sum(user2[movie] ** 2 for movie in common_movies) ** 0.5

    if norm_user1 == 0 or norm_user2 == 0:
        return 0

    return dot_product / (norm_user1 * norm_user2)

def pearson_correlation(user1, user2):
    common_movies = set(user1.keys()).intersection(set(user2.keys()))
    if not common_movies:
        return 0

    user1_ratings = np.array([user1[movie] for movie in common_movies])
    user2_ratings = np.array([user2[movie] for movie in common_movies])
    
    if len(user1_ratings) < 2 or len(user2_ratings) < 2:
        return 0

    mean_user1 = np.mean(user1_ratings)
    mean_user2 = np.mean(user2_ratings)
    
    centered_user1 = user1_ratings - mean_user1
    centered_user2 = user2_ratings - mean_user2
    
    numerator = np.sum(centered_user1 * centered_user2)
    denominator = np.sqrt(np.sum(centered_user1 ** 2)) * np.sqrt(np.sum(centered_user2 ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def adjusted_cosine_similarity(movie1, movie2, train_data, movie_count, total_users):
    common_users = [user for user in train_data if movie1 in train_data[user] and movie2 in train_data[user]]
    if not common_users:
        return 0

    movie1_ratings = np.array([train_data[user][movie1] for user in common_users])
    movie2_ratings = np.array([train_data[user][movie2] for user in common_users])
    user_means = np.array([np.mean(list(train_data[user].values())) for user in common_users])

    movie1_centered = movie1_ratings - user_means
    movie2_centered = movie2_ratings - user_means

    iuf_movie1_centered = apply_iuf(dict(zip(common_users, movie1_centered)), movie_count, total_users)
    iuf_movie2_centered = apply_iuf(dict(zip(common_users, movie2_centered)), movie_count, total_users)

    movie1_centered = np.array(list(iuf_movie1_centered.values()))
    movie2_centered = np.array(list(iuf_movie2_centered.values()))

    numerator = np.sum(movie1_centered * movie2_centered)
    denominator = np.sqrt(np.sum(movie1_centered ** 2)) * np.sqrt(np.sum(movie2_centered ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def predict_rating(train_data, movie_count, test_data, target_user, target_movie, similarity_func, total_users, similarity_threshold=0, amplification_factor=2.5, top_k=10):
    similarities = []
    target_user_ratings = {movie: rating for (user, movie, rating) in test_data if user == target_user and rating != 0}
    target_user_avg = np.mean(list(target_user_ratings.values())) if target_user_ratings else 0

    iuf_target_user_ratings = apply_iuf(target_user_ratings, movie_count, total_users)

    for user, ratings in train_data.items():
        if user != target_user and target_movie in ratings:
            iuf_ratings = apply_iuf(ratings, movie_count, total_users)
            similarity = similarity_func(iuf_target_user_ratings, iuf_ratings)
            if abs(similarity) >= similarity_threshold:  
                similarity = np.sign(similarity) * (abs(similarity) ** amplification_factor)  

                if similarity != 0:  
                    if similarity_func == pearson_correlation:
                        user_avg = np.mean(list(ratings.values()))
                        adjusted_rating = ratings[target_movie] - user_avg
                        similarities.append((similarity, adjusted_rating))
                    else:
                        similarities.append((similarity, ratings[target_movie]))

    if not similarities:
        return 3  

    similarities.sort(reverse=True, key=lambda x: abs(x[0]))  
    top_k_similarities = similarities[:top_k] 

    numerator = sum(sim * rating for sim, rating in top_k_similarities)
    denominator = sum(abs(sim) for sim, rating in top_k_similarities)

    if denominator == 0:
        return 3 

    prediction = target_user_avg + (numerator / denominator) if similarity_func == pearson_correlation else numerator / denominator
    prediction = max(1, min(prediction, 5)) 

    return prediction

def predict_item_rating(train_data, movie_count, test_data, target_user, target_movie, similarity_func, total_users, amplification_factor=2.5, top_k=10):
    similarities = []
    user_ratings = {movie: rating for (user, movie, rating) in test_data if user == target_user and rating != 0}

    for movie, rating in user_ratings.items():
        if movie != target_movie:
            similarity = similarity_func(target_movie, movie, train_data, movie_count, total_users)
            if similarity != 0:
                similarity = np.sign(similarity) * (abs(similarity) ** amplification_factor)
                similarities.append((similarity, rating))

    if not similarities:
        return 3  

    similarities.sort(reverse=True, key=lambda x: abs(x[0]))  
    top_k_similarities = similarities[:top_k]  

    numerator = sum(sim * rating for sim, rating in top_k_similarities)
    denominator = sum(abs(sim) for sim, rating in top_k_similarities)

    if denominator == 0:
        return 3 

    prediction = numerator / denominator
    prediction = max(1, min(prediction, 5))  

    return prediction

def user_based_cf(train_data, movie_count, test_data, similarity_func, total_users):
    predictions = []
    output_data = []

    for user, movie, true_rating in test_data:
        if true_rating == 0:
            predicted_rating = predict_rating(train_data, movie_count, test_data, user, movie, similarity_func, total_users)
            predictions.append((user, movie, predicted_rating))

    return predictions

def item_based_cf(train_data, movie_count, test_data, similarity_func, total_users):
    predictions = []

    for user, movie, true_rating in test_data:
        if true_rating == 0:
            predicted_rating = predict_item_rating(train_data, movie_count, test_data, user, movie, similarity_func, total_users)
            predictions.append((user, movie, predicted_rating))

    return predictions

def variance_based_cf(train_data, movie_count, movie_variance, test_data, total_users):
    predictions = []

    for user, movie, true_rating in test_data:
        if true_rating == 0:
            target_user_ratings = {movie: rating for (u, movie, rating) in test_data if u == user and rating != 0}
            var_adjusted_target_user_ratings = apply_variance_adjustment(target_user_ratings, movie_variance)

            similarities = []
            for u, ratings in train_data.items():
                if u != user and movie in ratings:
                    var_adjusted_ratings = apply_variance_adjustment(ratings, movie_variance)
                    similarity = pearson_correlation(var_adjusted_target_user_ratings, var_adjusted_ratings)
                    if abs(similarity) > 0:
                        user_avg = np.mean(list(ratings.values()))
                        adjusted_rating = ratings[movie] - user_avg
                        similarities.append((similarity, adjusted_rating))

            if not similarities:
                predictions.append((user, movie, 4))  
            else:
                similarities.sort(reverse=True, key=lambda x: abs(x[0]))
                top_k_similarities = similarities[:20]  

                numerator = sum(sim * rating for sim, rating in top_k_similarities)
                denominator = sum(abs(sim) for sim, rating in top_k_similarities)

                if denominator == 0:
                    predictions.append((user, movie, 4))  
                else:
                    target_user_avg = np.mean(list(target_user_ratings.values())) if target_user_ratings else 0
                    prediction = target_user_avg + (numerator / denominator)
                    prediction = max(1, min(prediction, 5))
                    predictions.append((user, movie, prediction))

    return predictions


def save_predictions(output_data, filename):
    with open(filename, 'w') as file:
        for user, movie, pred in output_data:
            file.write(f"{user} {movie} {pred}\n")



def average_predictions(pred_cosine, pred_pearson, pred_item, pred_variance):
    averaged_predictions = []
    for (user1, movie1, rating1), (user2, movie2, rating2), (user3, movie3, rating3), (user4, movie4, rating4) in zip(pred_cosine, pred_pearson, pred_item, pred_variance):
        assert user1 == user2 == user3 == user4
        assert movie1 == movie2 == movie3 == movie4
        average_rating = int(round(np.mean([rating1, rating2, rating3, rating4])))
        averaged_predictions.append((user1, movie1, average_rating))
    return averaged_predictions


def average_predictions_three_methods(pred_cosine, pred_pearson, pred_variance):
    averaged_predictions = []
    for (user1, movie1, rating1), (user2, movie2, rating2), (user3, movie3, rating3) in zip(pred_cosine, pred_pearson, pred_variance):
        assert user1 == user2 == user3 
        assert movie1 == movie2 == movie3 
        average_rating = int(round(np.mean([rating1, rating2, rating3])))
        averaged_predictions.append((user1, movie1, average_rating))
    return averaged_predictions

def average_predictions_one_method(method):
    averaged_predictions = []
    for (user1, movie1, rating1) in zip(method):
        assert user1
        assert movie1
        average_rating = int(round(np.mean([rating1])))
        averaged_predictions.append((user1, movie1, average_rating))
    return averaged_predictions


train_data, movie_count,movie_ratings = load_training_data('train.txt')
test_datasets = {
    'test5': load_test_data('test5.txt'),
    'test10': load_test_data('test10.txt'),
    'test20': load_test_data('test20.txt'),
}

total_users = len(train_data)
movie_variance = calculate_variance(movie_ratings)

for test_name, test_data in test_datasets.items():
    pred_cosine = user_based_cf(train_data, movie_count, test_data, cosine_similarity, total_users)
    pred_pearson = user_based_cf(train_data, movie_count, test_data, pearson_correlation, total_users)
    pred_item_based = item_based_cf(train_data, movie_count, test_data, adjusted_cosine_similarity, total_users)
    pred_variance_based = variance_based_cf(train_data, movie_count, movie_variance, test_data, total_users)

    # Excluding item_based actually gave me better results in the end.
    averaged_preds = average_predictions_three_methods(pred_cosine, pred_pearson, pred_variance_based)


    save_predictions(averaged_preds, f'averaged_{test_name}.txt')
