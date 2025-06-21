import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Train model (unchanged)
def train_model():
    print("Training started...")
    df = pd.read_csv('/Users/priyaharshitasingh/Desktop/homify_ml/DATASET.csv')  # Updated path
    df = df.sample(200, random_state=42)
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '')
    
    if 'wakeup_time' in df.columns:
        df['wakeup_min'] = df['wakeup_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    else:
        raise ValueError("Column 'wakeup_time' not found.")
    
    if 'cleanliness' in df.columns:
        cleanliness_map = {'messy': 0, 'average': 1, 'very_clean': 2}
        df['cleanliness_num'] = df['cleanliness'].str.lower().map(cleanliness_map)
        if df['cleanliness_num'].isnull().any():
            raise ValueError("Some values in 'cleanliness' could not be mapped.")
    else:
        raise ValueError("Column 'cleanliness' not found.")
    
    scaler = MinMaxScaler()
    num = scaler.fit_transform(df[['cleanliness_num', 'wakeup_min']])
    
    cat_cols = [
        'sleep_schedule', 'noise_tolerance', 'smoking', 'drinking',
        'exercise_frequency', 'diet', 'cooking_habits',
        'food_sharing', 'pet_friendly', 'guest_policy',
        'partying_at_home', 'overnight_guests'
    ]
    missing_cats = [col for col in cat_cols if col not in df.columns]
    if missing_cats:
        raise ValueError(f"Missing categorical columns: {missing_cats}")
    
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat = enc.fit_transform(df[cat_cols])
    
    X_all = np.hstack([num, cat])
    
    print("Generating training pairs...")
    pairs = []
    n = len(X_all)
    for i in range(n):
        if i % 100 == 0:
            print(f"Processing row {i} of {n}")
        for j in range(i+1, n):
            sim = cosine_similarity([X_all[i]], [X_all[j]])[0, 0]
            pairs.append((i, j, (sim + 1) / 2))
    
    pairs_df = pd.DataFrame(pairs, columns=['i', 'j', 'y'])
    
    def build_features(i, j):
        return np.abs(X_all[i] - X_all[j])
    
    pairs_df['features'] = pairs_df.apply(lambda row: build_features(int(row.i), int(row.j)), axis=1)
    X = np.vstack(pairs_df['features'])
    y = pairs_df['y'].values
    
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("R² Score:", score)
    
    return df, scaler, enc, model, X_all

# Compute compatibility scores (unchanged)
def compute_compatibility(user_data, df, scaler, enc, model, X_all):
    user_df = pd.DataFrame([user_data])
    
    if 'wakeup_time' in user_df.columns:
        user_df['wakeup_min'] = user_df['wakeup_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    else:
        raise ValueError("User data missing 'wakeup_time'.")
    
    if 'cleanliness' in user_df.columns:
        cleanliness_map = {'messy': 0, 'average': 1, 'very_clean': 2}
        user_df['cleanliness_num'] = user_df['cleanliness'].str.lower().map(cleanliness_map)
        if user_df['cleanliness_num'].isnull().any():
            raise ValueError("Invalid 'cleanliness' value in user data.")
    else:
        raise ValueError("User data missing 'cleanliness'.")
    
    user_num = scaler.transform(user_df[['cleanliness_num', 'wakeup_min']])
    
    cat_cols = [
        'sleep_schedule', 'noise_tolerance', 'smoking', 'drinking',
        'exercise_frequency', 'diet', 'cooking_habits',
        'food_sharing', 'pet_friendly', 'guest_policy',
        'partying_at_home', 'overnight_guests'
    ]
    user_cat = enc.transform(user_df[cat_cols])
    
    user_X = np.hstack([user_num, user_cat])
    
    scores = []
    for i in range(len(X_all)):
        feature_diff = np.abs(user_X[0] - X_all[i])
        sim_score = model.predict([feature_diff])[0]
        compatibility = sim_score * 100
        scores.append((i, max(0, min(100, compatibility))))
    
    return scores

# Flask routes
@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    print("STEP 1: Form submission received")
    try:
        print("STEP 2: Collecting form data")
        user_data = {
            'name': request.form.get('name'),
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'profession': request.form.get('profession'),
            'monthly_budget': request.form.get('monthly_budget'),
            'preferred_city': request.form.get('preferred_city'),
            'cleanliness': request.form.get('cleanliness', '').lower().replace('✨ i sparkle like a swiffer', 'very_clean').replace('🧼 tidy-ish, not monica geller', 'average').replace('🎨 organized chaos is my vibe', 'messy'),
            'sleep_schedule': request.form.get('sleep_schedule', '').lower().replace('🌅 early bird, rise & grind', 'early_bird').replace('🕰️ go with the flow', 'flexible').replace('🌙 vampire hours enthusiast', 'night_owl'),
            'wakeup_time': request.form.get('wakeup_time'),
            'noise_tolerance': request.form.get('noise_tolerance', '').lower().replace('🔕 peace & quiet please', 'low').replace('🎶 chill with tunes & convo', 'medium').replace('🔊 let’s make some noise', 'high'),
            'smoking': request.form.get('smoking', '').lower().replace('🚭 nope', 'no').replace('🔥 smoker\'s lounge status', 'yes'),
            'drinking': request.form.get('drinking', '').lower().replace('🥤 never touch the stuff', 'never').replace('🍷 only with the squad', 'occasionally').replace('🍻 happy hour is sacred', 'often'),
            'exercise_frequency': request.form.get('exercise_frequency', '').lower().replace('💪 gym rat', 'daily').replace('🧘‍♀️ occasionally active', 'sometimes').replace('🚫 exercise? i thought you said extra fries', 'never'),
            'diet': request.form.get('diet', '').lower().replace('🥕 plant-powered (veg)', 'vegetarian').replace('🍗 carnivore vibes', 'non_vegetarian').replace('🌱 hardcore vegan', 'vegan'),
            'cooking_habits': request.form.get('cooking_habits', '').lower().replace('👨‍🍳 masterchef in the making', 'often').replace('🥄 i cook when ubereats says no', 'sometimes').replace('🔥 kitchen = danger zone', 'never'),
            'food_sharing': request.form.get('food_sharing', '').lower().replace('🍕 what’s mine is yours', 'yes').replace('🙅‍♀️ hands off my snacks', 'no'),
            'pet_friendly': request.form.get('pet_friendly', '').lower().replace('🐶 love all fur babies', 'yes').replace('😬 allergic or just not my thing', 'no'),
            'guest_policy': request.form.get('guest_policy', '').lower().replace('🔐 my space = my sanctuary', 'never').replace('🎈 occasional hangouts', 'sometimes').replace('🏠 open house always', 'often'),
            'partying_at_home': request.form.get('partying_at_home', '').lower().replace('📚 more chill, less thrill', 'never').replace('🕺 when the vibe is right', 'sometimes').replace('🎉 party central', 'often'),
            'overnight_guests': request.form.get('overnight_guests', '').lower().replace('🛏️ sure, guests welcome', 'occasionally').replace('🚫 no sleepovers, please', 'never')
        }
        print("STEP 3: Form data collected:", user_data)

        print("STEP 4: Validating wakeup_time")
        if not user_data['wakeup_time']:
            return "Missing wakeup time.", 400
        try:
            hours, minutes = map(int, user_data['wakeup_time'].split(':'))
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                return "Invalid wakeup time. Use HH:MM (0–23 hours, 0–59 minutes).", 400
        except ValueError:
            return "Invalid wakeup time format. Use HH:MM (e.g., 08:30).", 400

        print("STEP 5: Validating required fields")
        required_fields = ['cleanliness', 'sleep_schedule', 'wakeup_time', 'noise_tolerance', 'smoking',
                           'drinking', 'exercise_frequency', 'diet', 'cooking_habits', 'food_sharing',
                           'pet_friendly', 'guest_policy', 'partying_at_home', 'overnight_guests']
        for field in required_fields:
            if not user_data[field]:
                return f"Missing required field: {field}", 400

        print("STEP 6: Validating monthly_budget")
        try:
            user_data['monthly_budget'] = float(user_data['monthly_budget'] or 0)
        except (ValueError, TypeError):
            return "Invalid monthly budget. Must be a number.", 400

        print("STEP 7: Saving to CSV")
        user_df = pd.DataFrame([user_data])
        csv_path = '/Users/priyaharshitasingh/Desktop/homify_ml/user_data.csv'
        user_df.to_csv(csv_path, index=False)
        print("STEP 8: CSV saved")

        print("STEP 9: Training model")
        df, scaler, enc, model, X_all = train_model()
        print("STEP 10: Model trained")
        df['monthly_budget'] = pd.to_numeric(df['monthly_budget'], errors='coerce').fillna(0)
        print("STEP 11: Computing compatibility scores")
        compatibility_scores = compute_compatibility(user_data, df, scaler, enc, model, X_all)
        print("STEP 12: Scores computed:", compatibility_scores[:5])
        profiles = df.reset_index().to_dict('records')
        print("STEP 13: Rendering ListView.html")
        return render_template('ListView.html', scores=compatibility_scores, profiles=profiles, name=user_data['name'])
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)