import streamlit as st
import random
import pandas as pd
import json
from pathlib import Path
from PIL import Image

# App configuration
st.set_page_config(page_title="Image Authenticity Test", layout="wide")

# Title and instructions
st.title("Which image is real?")
st.markdown("""
### Instructions:
1. You'll see two images of laser printed codes on packages side by side
2. One is a real photograph, the other is synthetically generated
3. Click on the image you believe is the real photograph
4. After submitting, you'll see if you were correct
5. We'll show you 10 comparisons in total
""")

# Initialize session state for tracking progress and results
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
    st.session_state.results = []
    st.session_state.correct_answers = 0
    st.session_state.method1_mistakes = 0
    st.session_state.method2_mistakes = 0

# Sample data setup (replace with your actual image paths)
def setup_image_data():
    # This should point to your directories containing images
    real_images = list(Path("data/real").glob("*.png"))
    method1_synth = list(Path("data/simulated").glob("*.png"))
    method2_synth = list(Path("data/sdXL").glob("*.png"))
    
    # Create pairs for comparison
    comparisons = []
    for real_img in real_images[:10]:  # Use first 10 real images
        # Randomly choose which method to compare against
        if random.random() > 0.5:
            synth_img = random.choice(method1_synth)
            method = "Method 1"
        else:
            synth_img = random.choice(method2_synth)
            method = "Method 2"
        
        # Randomize left/right position
        if random.random() > 0.5:
            left_img, right_img = real_img, synth_img
            correct = "left"
        else:
            left_img, right_img = synth_img, real_img
            correct = "right"
            
        comparisons.append({
            "left": left_img,
            "right": right_img,
            "correct": correct,
            "method": method
        })
    
    return comparisons

# Load or create the comparisons
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = setup_image_data()
    # check if it works
    if not st.session_state.comparisons:
        st.error("No images found. Please check your image directories.")
        st.stop()

# Display current comparison
if st.session_state.current_round < len(st.session_state.comparisons):
    current = st.session_state.comparisons[st.session_state.current_round]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(Image.open(current["left"]), use_container_width =True)
        if st.button("Choose Left", key="left"):
            if current["correct"] == "left":
                st.session_state.correct_answers += 1
                st.success("Correct! This was the real image.")
            else:
                st.error("Incorrect. This was the synthetic image.")
                # Track which method fooled the user
                if current["method"] == "Method 1":
                    st.session_state.method1_mistakes += 1
                else:
                    st.session_state.method2_mistakes += 1
                
            st.session_state.results.append({
                "round": st.session_state.current_round + 1,
                "choice": "left",
                "correct": current["correct"] == "left",
                "method": current["method"]
            })
            st.session_state.current_round += 1
            st.rerun()
    
    with col2:
        st.image(Image.open(current["right"]), use_container_width =True)
        if st.button("Choose Right", key="right"):
            if current["correct"] == "right":
                st.session_state.correct_answers += 1
                st.success("Correct! This was the real image.")
            else:
                st.error("Incorrect. This was the synthetic image.")
                # Track which method fooled the user
                if current["method"] == "Method 1":
                    st.session_state.method1_mistakes += 1
                else:
                    st.session_state.method2_mistakes += 1
                
            st.session_state.results.append({
                "round": st.session_state.current_round + 1,
                "choice": "right",
                "correct": current["correct"] == "right",
                "method": current["method"]
            })
            st.session_state.current_round += 1
            st.rerun()

# Show results after all rounds
# Show results after all rounds
else:
    st.balloons()
    st.success("Test completed! Thank you for participating.")
    
    # Calculate statistics
    total_rounds = len(st.session_state.comparisons)
    accuracy = (st.session_state.correct_answers / total_rounds) * 100
    
    st.subheader("Your Results")
    st.write(f"Correct answers: {st.session_state.correct_answers}/{total_rounds} ({accuracy:.1f}%)")
    
    # Method comparison
    st.subheader("Method Comparison")
    st.write("These results show which synthetic generation method was more often mistaken for real:")
    
    method1_percentage = (st.session_state.method1_mistakes / total_rounds) * 100 if total_rounds > 0 else 0
    method2_percentage = (st.session_state.method2_mistakes / total_rounds) * 100 if total_rounds > 0 else 0
    
    st.write("Pixel Manipulation (Method 1)", f"{st.session_state.method1_mistakes} times mistaken for real ({method1_percentage:.1f}%)")
    st.write("Stable Diffusion XL (Method 2)", f"{st.session_state.method2_mistakes} times mistaken for real ({method2_percentage:.1f}%)")

    # Prepare the results data structure
    json_path = Path("data.json")
    
    results_data = {
        "correct_answers": st.session_state.correct_answers,
        "total_rounds": total_rounds,
        "accuracy": accuracy,
        "method1_mistakes": st.session_state.method1_mistakes,
        "method2_mistakes": st.session_state.method2_mistakes,
        "detailed_results": st.session_state.results
    }

    # Check if file exists and load existing data
    existing_data = []
    if json_path.exists():
        with open(json_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    
    # Append new results
    existing_data.append(results_data)
    
    # Save back to file
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=4)
    
    st.success("Results saved successfully!")
    
    # Show detailed results - only one checkbox needed
    if st.checkbox("Show detailed results", key="detailed_results_checkbox"):
        st.table(pd.DataFrame(st.session_state.results))
    
    # Option to restart
    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()