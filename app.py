import numpy as np
import pandas as pd
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

num_responses = 1000

# Helper functions for multi-select (association rule mining columns)
def multi_select(options, min_items=1, max_items=None, p=None):
    if max_items is None:
        max_items = len(options)
    n = np.random.randint(min_items, max_items+1)
    # weighted random sample if probabilities given
    if p is not None:
        selected = np.random.choice(options, size=n, replace=False, p=p)
    else:
        selected = np.random.choice(options, size=n, replace=False)
    return ", ".join(selected)

# 1. Age: Skewed distribution + outliers
ages = np.concatenate([
    np.random.normal(loc=30, scale=8, size=int(num_responses * 0.93)).astype(int),
    np.random.normal(loc=55, scale=5, size=int(num_responses * 0.07)).astype(int)
])
ages = np.clip(ages, 15, 70)

# 2. Gender (Imbalanced as in real life app data)
genders = np.random.choice(['Male', 'Female', 'Other', 'Prefer not to say'], size=num_responses, p=[0.52, 0.45, 0.02, 0.01])

# 3. City Type (Metro most common, then urban, few rural)
city_types = np.random.choice(['Metro', 'Urban', 'Semi-Urban', 'Rural'], size=num_responses, p=[0.45, 0.32, 0.13, 0.10])

# 4. Annual Income (Log-normal for skew, add outliers)
incomes = np.random.lognormal(mean=11.1, sigma=0.5, size=num_responses).astype(int)  # Local currency, e.g., INR

# Introduce some extreme outliers
outlier_indices = np.random.choice(num_responses, size=8, replace=False)
incomes[outlier_indices] = incomes[outlier_indices] * np.random.randint(2, 6, size=8)

# 5. Education
edu_levels = ['High School', 'Graduate', 'Postgraduate', 'Doctorate', 'Other']
edu_probs = [0.12, 0.56, 0.27, 0.03, 0.02]
education = np.random.choice(edu_levels, size=num_responses, p=edu_probs)

# 6. Occupation
occupations = ['Student', 'Professional', 'Homemaker', 'Retired', 'Other']
occupation_probs = [0.13, 0.65, 0.12, 0.06, 0.04]
occupation = np.random.choice(occupations, size=num_responses, p=occupation_probs)

# 7. Steps per day (numerical, skewed with outliers)
steps = np.concatenate([
    np.random.normal(7000, 2500, int(num_responses * 0.92)),
    np.random.normal(14000, 1000, int(num_responses * 0.08))
]).astype(int)
steps = np.clip(steps, 500, 25000)

# 8. Workout days per week (skewed toward 0-3 and 5-7)
workout_days = np.random.choice([0,1,2,3,4,5,6,7], size=num_responses, p=[0.09, 0.15, 0.17, 0.19, 0.14, 0.11, 0.08, 0.07])

# 9. Preferred workout type
workout_types = ['Cardio', 'Strength Training', 'Yoga', 'Sports', 'Mixed', 'None']
workout_type = np.random.choice(workout_types, size=num_responses, p=[0.31, 0.23, 0.17, 0.07, 0.14, 0.08])

# 10. Average workout session duration (minutes, right skewed)
durations = np.random.gamma(shape=2.5, scale=16, size=num_responses).astype(int)
durations = np.clip(durations, 5, 180)

# 11. Calories burned/day (correlate loosely with steps and workouts)
calories = (steps * 0.045 + durations * workout_days * 1.2 + np.random.normal(150, 100, num_responses)).astype(int)
calories = np.clip(calories, 400, 3000)

# 12. Uses fitness app
uses_app = np.random.choice(['Yes', 'No'], size=num_responses, p=[0.82, 0.18])

# 13. Features used (multi-select, only if uses app)
feature_options = ['Step Tracker', 'Workout Plans', 'Diet Tracking', 'Social Community', 'Challenges', 'Progress Tracking', 'None']
features_used = [
    multi_select(feature_options[:-1], min_items=1, max_items=4, p=[0.25,0.18,0.18,0.10,0.13,0.13]) if uses_app[i]=='Yes' else "None"
    for i in range(num_responses)
]

# 14. App usage frequency
freq_options = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
usage_freq = [
    np.random.choice(freq_options, p=[0.69,0.15,0.08,0.05,0.03]) if uses_app[i]=='Yes' else 'Never'
    for i in range(num_responses)
]

# 15. Subscribed to premium app
subscribed_premium = []
for i in range(num_responses):
    if uses_app[i] == 'No' or usage_freq[i] in ['Rarely', 'Never']:
        subscribed_premium.append('No')
    else:
        # Higher chance to subscribe if high steps, workout days, calories, higher income
        base_prob = 0.14
        if steps[i]>11000: base_prob += 0.14
        if workout_days[i]>4: base_prob += 0.11
        if incomes[i]>2500000: base_prob += 0.08
        if calories[i]>1800: base_prob += 0.08
        if education[i] in ['Graduate','Postgraduate','Doctorate']: base_prob += 0.03
        subscribed_premium.append('Yes' if random.random()<base_prob else 'No')

# 16. Motivators for upgrading (multi-select, association mining format)
motivation_options = ['Personalized Plans', 'Ad-free Experience', 'Advanced Analytics', '1:1 Coach', 'Social Competitions', 'Custom Diet Plans', 'None']
motivators = [
    multi_select(motivation_options[:-1], min_items=1, max_items=3) if uses_app[i]=='Yes' else 'None'
    for i in range(num_responses)
]

# 17. Willingness to pay per month (numerical, right skew, higher for premium subscribers)
willingness_pay = []
for i in range(num_responses):
    if subscribed_premium[i]=='Yes':
        amount = np.random.normal(700, 150)  # e.g., INR
        if incomes[i]>2500000:
            amount += np.random.normal(200, 80)
    else:
        amount = np.random.normal(350, 120)
    # outliers
    if random.random()<0.015:
        amount += np.random.normal(2000, 600)
    willingness_pay.append(max(50, int(abs(amount))))

# 18. Fitness importance (1-10, higher for premium, free/rare users have more low scores)
fitness_importance = []
for i in range(num_responses):
    if subscribed_premium[i]=='Yes':
        val = int(np.random.normal(8, 1.3))
    elif uses_app[i]=='No':
        val = int(np.random.normal(4, 1.2))
    else:
        val = int(np.random.normal(6, 1.6))
    fitness_importance.append(np.clip(val,1,10))

# 19. Biggest challenge
challenge_options = ['Lack of Time', 'Motivation', 'Guidance', 'Cost', 'Health Issues', 'Other']
challenge = np.random.choice(challenge_options, size=num_responses, p=[0.27,0.22,0.18,0.14,0.11,0.08])

# 20. Who motivates most
motivator_options = ['Family', 'Friends', 'Influencers', 'Self', 'None']
motivated_by = np.random.choice(motivator_options, size=num_responses, p=[0.29,0.18,0.09,0.41,0.03])

# 21. Device for fitness tracking
device_options = ['Smartphone', 'Fitness Band', 'Smartwatch', 'Manual Entry', 'None']
device = np.random.choice(device_options, size=num_responses, p=[0.58, 0.17, 0.18, 0.05, 0.02])

# 22. Likelihood to recommend app (NPS 1-10)
nps = []
for i in range(num_responses):
    if uses_app[i]=='Yes' and subscribed_premium[i]=='Yes':
        score = int(np.random.normal(8, 1.4))
    elif uses_app[i]=='Yes':
        score = int(np.random.normal(7, 1.8))
    else:
        score = int(np.random.normal(4, 1.3))
    nps.append(np.clip(score, 1, 10))

# 23. Engaged in-app content
content_options = ['Video Workouts', 'Articles', 'Community Posts', 'Live Classes', 'Challenges']
content_engaged = [np.random.choice(content_options) if uses_app[i]=='Yes' else 'None' for i in range(num_responses)]

# 24. Interested in group challenges
group_challenges = [np.random.choice(['Yes', 'No'], p=[0.63,0.37]) if uses_app[i]=='Yes' else 'No' for i in range(num_responses)]

# 25. Fitness goal
goal_options = ['Weight Loss', 'Muscle Gain', 'General Wellness', 'Athletic Training', 'Rehabilitation', 'None']
goals = np.random.choice(goal_options, size=num_responses, p=[0.36, 0.21, 0.27, 0.08, 0.06, 0.02])

# Construct DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'City_Type': city_types,
    'Annual_Income': incomes,
    'Education_Level': education,
    'Occupation': occupation,
    'Steps_Per_Day': steps,
    'Workout_Days_Per_Week': workout_days,
    'Preferred_Workout_Type': workout_type,
    'Workout_Duration_Min': durations,
    'Calories_Burned_Per_Day': calories,
    'Uses_Fitness_App': uses_app,
    'Features_Used': features_used,                # Association Rule Mining format
    'App_Usage_Frequency': usage_freq,
    'Subscribed_Premium': subscribed_premium,
    'Motivators_For_Upgrade': motivators,          # Association Rule Mining format
    'Willingness_To_Pay': willingness_pay,
    'Fitness_Importance_1_10': fitness_importance,
    'Biggest_Challenge': challenge,
    'Fitness_Motivated_By': motivated_by,
    'Preferred_Device': device,
    'NPS_Recommend_1_10': nps,
    'Content_Engaged': content_engaged,
    'Interested_In_Group_Challenges': group_challenges,
    'Fitness_Goal': goals,
})

# Save to CSV
df.to_csv("synthetic_fitness_survey.csv", index=False)

print("Sample of generated data:\n")
print(df.head(10))
print("\nDataset shape:", df.shape)
