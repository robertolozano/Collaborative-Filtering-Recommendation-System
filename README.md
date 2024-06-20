# Collaborative Filtering Recommendation System

A collaborative filtering recommendation system using user-based, item-based, and variance-adjusted methods. Implements cosine similarity, Pearson correlation, and adjusted cosine similarity algorithms to predict movie ratings. Combines multiple methods for accurate recommendations.

## Approach

1. Data Loading:

   - Load training and test data files containing user, movie, and rating information.

2. Similarity Calculations:

   - Calculate similarities between users or movies using cosine similarity, Pearson correlation, and adjusted cosine similarity.

3. Rating Predictions:

   - Use user-based and item-based collaborative filtering to predict ratings.
   - Apply variance adjustments to account for rating inconsistencies.

4. Inverse User Frequency (IUF) Adjustment:

   - Apply IUF to adjust ratings based on the frequency of movie ratings across all users.

5. Prediction Averaging:

   - Combine predictions from different methods (user-based, item-based, and variance-based) to improve accuracy.

6. Save Predictions:
   - Save the averaged predictions to output files for further analysis.

## Features

- **User-Based Filtering**: Predict ratings based on user-user similarities.
- **Item-Based Filtering**: Predict ratings based on item-item similarities.
- **Variance Adjustment**: Adjust ratings based on variance to improve accuracy.
- **Multiple Similarity Metrics**: Supports cosine similarity, Pearson correlation, and adjusted cosine similarity.
- **Prediction Averaging**: Combines predictions from different methods for enhanced accuracy.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy

### Usage

1. **Place Data Files**: Place the training and test data files (`train.txt`, `test5.txt`, `test10.txt`, `test20.txt`) in the same directory as the script.

2. **Run the Script**: Execute the script to generate movie rating predictions. The results will be saved in files named `averaged_test5.txt`, `averaged_test10.txt`, and `averaged_test20.txt`.

### Example

```python
# Load training and test data
train_data, movie_count, movie_ratings = load_training_data('train.txt')
test_datasets = {
    'test5': load_test_data('test5.txt'),
    'test10': load_test_data('test10.txt'),
    'test20': load_test_data('test20.txt'),
}

# Calculate total users and movie variance
total_users = len(train_data)
movie_variance = calculate_variance(movie_ratings)

# Generate and save predictions
for test_name, test_data in test_datasets.items():
    pred_cosine = user_based_cf(train_data, movie_count, test_data, cosine_similarity, total_users)
    pred_pearson = user_based_cf(train_data, movie_count, test_data, pearson_correlation, total_users)
    pred_item_based = item_based_cf(train_data, movie_count, test_data, adjusted_cosine_similarity, total_users)
    pred_variance_based = variance_based_cf(train_data, movie_count, movie_variance, test_data, total_users)

    averaged_preds = average_predictions_three_methods(pred_cosine, pred_pearson, pred_variance_based)
    save_predictions(averaged_preds, f'averaged_{test_name}.txt')
```

## Functions

**load_training_data(file_path)**

- Loads the training data from the specified file path.

**load_test_data(file_path)**

- Loads the test data from the specified file path.

**apply_iuf(ratings, movie_count, total_users)**

- Applies the Inverse User Frequency (IUF) adjustment to the ratings.

**apply_variance_adjustment(ratings, movie_variance)**

- Applies variance adjustment to the ratings.

**calculate_variance(movie_ratings)**

- Calculates the variance of ratings for each movie.

**cosine_similarity(user1, user2)**

- Calculates the cosine similarity between two users.

**pearson_correlation(user1, user2)**

- Calculates the Pearson correlation between two users.

**adjusted_cosine_similarity(movie1, movie2, train_data, movie_count, total_users)**

- Calculates the adjusted cosine similarity between two movies.

**predict_rating(train_data, movie_count, test_data, target_user, target_movie, similarity_func, total_users, similarity_threshold=0, amplification_factor=2.5, top_k=10)**

- Predicts the rating for a target user and movie using the specified similarity function.

**predict_item_rating(train_data, movie_count, test_data, target_user, target_movie, similarity_func, total_users, amplification_factor=2.5, top_k=10)**

- Predicts the rating for a target user and movie using item-based collaborative filtering.

**user_based_cf(train_data, movie_count, test_data, similarity_func, total_users)**

- Generates predictions using user-based collaborative filtering.

**item_based_cf(train_data, movie_count, test_data, similarity_func, total_users)**

- Generates predictions using item-based collaborative filtering.

**variance_based_cf(train_data, movie_count, movie_variance, test_data, total_users)**

- Generates predictions using variance-based collaborative filtering.

**save_predictions(output_data, filename)**

- Saves the predictions to a file.

**average_predictions(pred_cosine, pred_pearson, pred_item, pred_variance)**

- Averages the predictions from multiple methods.

**average_predictions_three_methods(pred_cosine, pred_pearson, pred_variance)**

- Averages the predictions from three methods.

**average_predictions_one_method(method)**

- Averages the predictions from one method.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Inspired by collaborative filtering techniques in recommendation systems.
Utilizes classic similarity measures and variance adjustment for improved accuracy.
