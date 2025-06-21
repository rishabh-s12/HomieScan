from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import random

# Import the scoring logic from your repo
from score import compute_score  # make sure score.py is in the same directory

app = Flask(__name__)

# 🔌 MongoDB Connection
MONGO_URI = "mongodb+srv://mayurjindal905:LZXQWbsf1rGMalHB@homiescandb.yb6yolt.mongodb.net/Homiescan?retryWrites=true&w=majority&appName=HomieScanDb"
client = MongoClient(MONGO_URI)
db = client["Homiescan"]
collection = db["profiles"]

# 🧠 Agreement templates (partial for brevity; complete yours based on earlier response)
agreement_lines = {
    "sleep_schedule": {
        "early riser": "• At the first crow of dawn, peace shall be kept till coffee is consumed.",
        "night owl": "• Let there be silence till the stars fade; the owl must rest.",
        "flexible": "• Whether day or night, thou shalt respect thy roomie's slumber."
    },
    "diet": {
        "vegetarian": "• The beast shall not be cooked in our hallowed pan.",
        "non-vegetarian": "• Let meat be shared like bounty after battle.",
        "vegan": "• No moo juice or hen fruit shall enter this holy fridge."
    },
    "overnight_guests": {
        "yes": "• If thy guests stay the night, they owe us breakfast.",
        "no": "• Overnight stay be forbidden—nay, even for thine soulmate.",
        "occasionally": "• A rare sleepover be allowed, but not without prior treaty."
    },
    # Add all your feature mappings here...
}

# 🔮 Agreement Generator
def generate_agreement(preferences):
    selected = random.sample(list(preferences.keys()), 5)
    lines = []
    for key in selected:
        value = preferences.get(key)
        clause = agreement_lines.get(key, {}).get(value)
        if clause:
            lines.append(clause)
        else:
            lines.append(f"• No ancient scroll found for {key} = {value}.")
    return lines

# 🧪 Main Match Endpoint
@app.route("/roommate-match", methods=["POST"])
def roommate_match():
    data = request.get_json()
    user_id = data.get("user_id")
    match_id = data.get("match_id")

    if not user_id or not match_id:
        return jsonify({"error": "user_id and match_id are required"}), 400

    try:
        user = collection.find_one({"_id": ObjectId(user_id)})
        match = collection.find_one({"_id": ObjectId(match_id)})
    except Exception as e:
        return jsonify({"error": "Invalid ObjectId", "details": str(e)}), 400

    if not user or not match:
        return jsonify({"error": "User or match not found"}), 404

    # 🧠 Compatibility Score
    score = compute_score(user, match)

    # 📜 Generate Agreement from combined preferences
    agreement_prefs = {k: user.get(k) or match.get(k) for k in agreement_lines.keys()}
    agreement = generate_agreement(agreement_prefs)

    return jsonify({
        "compatibility_score": score,
        "agreement": agreement
    })

@app.route("/", methods=["GET"])
def home():
    return "🏠 HomieScan Roommate API is running!"

if __name__ == "__main__":
    app.run(debug=True)
