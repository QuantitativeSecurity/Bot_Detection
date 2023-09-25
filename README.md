Bot Detection System

This system is designed to detect and classify user accounts as either bots or genuine users based on given account features. The system uses a combination of supervised, unsupervised, and semi-supervised learning techniques to achieve this.
Features:

    Data Preprocessing: Preprocesses the input data by scaling and splitting it into training and test sets.
    Supervised Learning: Uses a Random Forest classifier to train on labeled data and predict whether an account is a bot.
    Unsupervised Learning: Applies KMeans clustering to classify user accounts into clusters.
    Semi-supervised Learning: Uses Label Spreading to predict class labels for unlabeled data points.
    Ensemble Prediction: Combines predictions from supervised, unsupervised, and semi-supervised methods to make final classifications.
    Action on Bots: For accounts predicted as bots, a placeholder function (ban_and_store_info) is provided to take action, like banning the account.

Usage:

    Ensure you have the required libraries installed, including pandas, numpy, and sklearn.
    Load your labeled dataset in the labeled_data.csv file.
    Run the script. The script will preprocess the data, apply the different learning techniques, and generate predictions.
    Based on the performance of the supervised classifier, the system will either use just the supervised classifier or an ensemble of all methods for bot detection.
    If an account is classified as a bot, the ban_and_store_info function is called for that account. Modify this function as per your needs to take appropriate actions on detected bots.

Code Structure:

    preprocess_data: Accepts a dataframe and returns training and test sets after scaling.
    select_best_classifier: Uses grid search to select the best parameters for the Random Forest classifier.
    apply_clustering: Performs KMeans clustering on the given data.
    apply_label_spreading: Applies the Label Spreading algorithm to predict labels for unlabeled instances.
    ensemble_predictions: Combines predictions from different methods using given weights.
    ban_and_store_info: Placeholder function to take action on detected bots.

Important Notes:

    The labeled_data.csv file should have columns: user_mentions_count, followers_to_friends_ratio, user_statuses_count, account_age_days, and is_bot.
    The is_bot column should have binary labels, with 1 indicating a bot and 0 indicating a genuine user.
    For unlabeled instances, the is_bot value should be -1.
    Adjust the satisfactory_threshold as per your requirements. If the supervised classifier's performance exceeds this threshold, only the supervised method will be used for predictions.
    The ban_and_store_info function is a placeholder. Replace it with your specific implementation for taking actions on detected bots.
