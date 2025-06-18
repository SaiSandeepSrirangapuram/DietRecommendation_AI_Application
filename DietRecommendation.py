import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('done_food_data.csv')
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    scaler = StandardScaler()
    features = ['Energy_kcal', 'Protein_g', 'Carb_g', 'Fat_g', 'category_encoded']
    X = scaler.fit_transform(df[features])
    nn = NearestNeighbors(n_neighbors=10)  # Use 10 neighbors first; we'll filter after
    nn.fit(X)
    return df, label_encoder, scaler, nn

df, label_encoder, scaler, nn = load_data()

# --- UI Improvements ---
st.title(" AI Diet Recommendation Tool")
st.markdown("Select your nutrient preferences and dietary goal to get healthy food suggestions!")
# st.image("https://cdn.pixabay.com/photo/2016/11/23/15/38/diet-1853291_960_720.jpg", use_column_width=True)

# Sidebar Inputs
st.sidebar.title("Your Nutrition Preferences")

energy = st.sidebar.slider("Calories (kcal)", 0, 1000, 250)
protein = st.sidebar.slider("Protein (g)", 0, 100, 15)
carbs = st.sidebar.slider("Carbohydrates (g)", 0, 100, 30)
fat = st.sidebar.slider("Fat (g)", 0, 100, 8)
category = st.sidebar.selectbox("Goal Category", list(label_encoder.classes_))

# --- Add Dietary Filter ---
veg_only = st.sidebar.checkbox("Vegetarian Only")

# Filter dataframe if vegetarian filter applied
df_filtered = df.copy()
if veg_only:
    # Adjust this filter according to your dataset's FoodGroup values
    df_filtered = df_filtered[df_filtered['FoodGroup'].str.contains('Vegetarian|Plant', case=False, na=False)]
    if df_filtered.empty:
        st.sidebar.warning("No vegetarian foods found in dataset!")

# --- Sort by User Goals ---
if category == "Weight_Loss":
    df_filtered = df_filtered[df_filtered['Energy_kcal'] < 300]
elif category == "Muscle_Gain":
    df_filtered = df_filtered[df_filtered['Protein_g'] > 20]
# Add more goal-based filters if you want

if df_filtered.empty:
    st.warning("No foods match your dietary filters and goal criteria. Please adjust your selections.")
else:
    # Re-encode and scale filtered data
    category_encoded = label_encoder.transform([category])[0]
    features = ['Energy_kcal', 'Protein_g', 'Carb_g', 'Fat_g', 'category_encoded']

    X_filtered = df_filtered.copy()
    X_filtered['category_encoded'] = category_encoded  # Set all to selected category for input scaling consistency

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered[features])

    # Fit NearestNeighbors on filtered data
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_scaled)

    # Process input
    input_df = pd.DataFrame([{
        'Energy_kcal': energy,
        'Protein_g': protein,
        'Carb_g': carbs,
        'Fat_g': fat,
        'category_encoded': category_encoded
    }])
    input_scaled = scaler.transform(input_df)

    # Find Recommendations
    distances, indices = nn.kneighbors(input_scaled)
    recommendations = df_filtered.iloc[indices[0]][['Descrip', 'FoodGroup', 'Energy_kcal', 'Protein_g', 'Carb_g', 'Fat_g', 'category']]

    st.subheader("Based on your preferences, here are some food suggestions:")
    st.table(recommendations.reset_index(drop=True))

    # --- User Feedback ---
    feedback = st.radio("Was this helpful?", ["Yes", "No"])
    if feedback == "Yes":
        st.success("Thanks! We'll remember your taste.")
        # Here you can add code to log this feedback for future personalization
    elif feedback == "No":
        st.info("Thanks for your feedback! We'll try to improve.")

