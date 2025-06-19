import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

print("Training started…")

# 1. Load your dataset (500 or 1000 profiles)
df = pd.read_csv('/Users/priyaharshitasingh/Downloads/DATASET.csv')
df = df.sample(200, random_state=42)
print("Columns in CSV:", df.columns.tolist())

# 2. Standardize column names for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '')

print("Standardized columns:", df.columns.tolist())

# 3. Prepare Features

# a) Convert wakeup_time to numeric minutes
if 'wakeup_time' in df.columns:
    df['wakeup_min'] = df['wakeup_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
else:
    raise ValueError("Column 'wakeup_time' not found in the dataset.")

# b) Map 'cleanliness' to numeric
if 'cleanliness' in df.columns:
    print("Unique cleanliness values:", df['cleanliness'].unique())
    cleanliness_map = {
        'messy': 0,
        'average': 1,
        'very_clean': 2
    }
    df['cleanliness_num'] = df['cleanliness'].str.lower().map(cleanliness_map)
    if df['cleanliness_num'].isnull().any():
        print("Unmapped cleanliness values:", df['cleanliness'].unique())
        raise ValueError("Some values in 'cleanliness' could not be mapped. Please update cleanliness_map.")
else:
    raise ValueError("Column 'cleanliness' not found in the dataset.")

# c) Scale numeric columns
scaler = MinMaxScaler()
num = scaler.fit_transform(df[['cleanliness_num', 'wakeup_min']])

# d) One-hot encode categorical columns
cat_cols = [
    'sleep_schedule', 'noise_tolerance', 'smoking', 'drinking',
    'exercise_frequency', 'diet', 'cooking_habits',
    'food_sharing', 'pet_friendly', 'guest_policy',
    'partying_at_home', 'overnight_guests'
]
missing_cats = [col for col in cat_cols if col not in df.columns]
if missing_cats:
    raise ValueError(f"The following categorical columns are missing from the dataset: {missing_cats}")

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat = enc.fit_transform(df[cat_cols])

X_all = np.hstack([num, cat])

print("Feature preparation complete. Generating training pairs...")

# 4. Create training data using pairwise combinations with progress prints
pairs = []
n = len(X_all)
for i in range(n):
    if i % 100 == 0:
        print(f"Processing row {i} of {n}")
    for j in range(i+1, n):
        sim = cosine_similarity([X_all[i]], [X_all[j]])[0, 0]
        pairs.append((i, j, (sim + 1) / 2))

pairs_df = pd.DataFrame(pairs, columns=['i', 'j', 'y'])

# 5. Build model-ready features
def build_features(i, j):
    return np.abs(X_all[i] - X_all[j])

pairs_df['features'] = pairs_df.apply(lambda row: build_features(int(row.i), int(row.j)), axis=1)
X = np.vstack(pairs_df['features'])
y = pairs_df['y'].values

print("Training/test split and model training starting...")

# 6. Train & evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("R² Score:", score)

# 7. Function to compute compatibility scores for a new user
def compute_compatibility(user_data, df, scaler, enc, model, X_all):
    """
    Compute compatibility scores between a new user and all profiles in the dataset.
    
    Parameters:
    - user_data: Dictionary with user’s data (same format as dataset columns)
    - df: Original dataset (for reference)
    - scaler: Fitted MinMaxScaler
    - enc: Fitted OneHotEncoder
    - model: Trained GradientBoostingRegressor
    - X_all: Preprocessed dataset features
    
    Returns:
    - List of tuples (index, compatibility_score) where score is in percentage
    """
    # Create a DataFrame for the user’s data
    user_df = pd.DataFrame([user_data])
    
    # Preprocess user data
    # a) Convert wakeup_time to minutes
    if 'wakeup_time' in user_df.columns:
        user_df['wakeup_min'] = user_df['wakeup_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    else:
        raise ValueError("User data missing 'wakeup_time'.")
    # b) Map cleanliness to numeric
    if 'cleanliness' in user_df.columns:
        user_df['cleanliness_num'] = user_df['cleanliness'].str.lower().map(cleanliness_map)
        if user_df['cleanliness_num'].isnull().any():
            raise ValueError("Invalid 'cleanliness' value in user data.")
    else:
        raise ValueError("User data missing 'cleanliness'.")
    
    # c) Scale numeric columns
    user_num = scaler.transform(user_df[['cleanliness_num', 'wakeup_min']])
    
    # d) One-hot encode categorical columns
    user_cat = enc.transform(user_df[cat_cols])
    
    # Combine features
    user_X = np.hstack([user_num, user_cat])
    
    # Compute compatibility scores
    scores = []
    for i in range(len(X_all)):
        # Calculate feature difference
        feature_diff = np.abs(user_X[0] - X_all[i])
        # Predict similarity score
        sim_score = model.predict([feature_diff])[0]
        # Convert to percentage (0–100)
        compatibility = sim_score * 100
        scores.append((i, max(0, min(100, compatibility)))) # Clip to 0–100
    
    return scores
# Example usage for a new user
if __name__ == "__main__":
    # Sample user data (adjust values based on your dataset’s format)
    sample_user = {
        'wakeup_time': '07:30',
        'cleanliness': 'very_clean',
        'sleep_schedule': 'early_bird',
        'noise_tolerance': 'low',
        'smoking': 'no',
        'drinking': 'occasionally',
        'exercise_frequency': 'weekly',
        'diet': 'vegetarian',
        'cooking_habits': 'often',
        'food_sharing': 'yes',
        'pet_friendly': 'yes',
        'guest_policy': 'sometimes',
        'partying_at_home': 'rarely',
        'overnight_guests': 'occasionally'
    }
    
    # Compute compatibility scores
    compatibility_scores = compute_compatibility(sample_user, df, scaler, enc, model, X_all)
    
    # Display results
    print("\nCompatibility Scores:")
    for idx, score in sorted(compatibility_scores, key=lambda x: x[1], reverse=True):
        print(f"Profile {idx}: {score:.2f}% compatibility")