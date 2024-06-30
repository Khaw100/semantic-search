import streamlit as st
from ModelClasses import *  # Import your model classes
from huggingface_hub import hf_hub_download
import pickle
import csv
from streamlit import radio  # Import the necessary component

# Set page configuration
st.set_page_config(page_title="EduLearning", page_icon=":books:", layout="centered")

# Load the models
repo_id = 'Wipiii/Semantic'
filename_allmpnet = 'sbert_unstemmed_model_allmpnet_v2.pxl'
filename_minilm = 'sbert_unstemmed_model_allmpnet_v2.pxl'
filename_mpnet = 'sbert_unstemmed_model_allmpnet_v2.pxl'
filename_ft_mpnet = 'sbert_unstemmed_model_allmpnet_v2.pxl'
filename_ft_allmpnet = 'ft_sbert_unstemmed_model_allmpnet_v2.pxl'
filename_ft_minilm = 'sbert_unstemmed_model_allmpnet_v2.pxl'

# Function to load the model
# @st.cache_resource
# def load_model(filename):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     file_path = hf_hub_download(repo_id=repo_id, filename=filename)
#     with open(file_path, 'rb') as f:
#         model = pickle.load(f)
    
#     return model
@st.cache_resource
def load_model(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = torch.load(file_path, map_location=device)
    
    return model

model_dict = {
    "allmpnet_v2": load_model(filename_allmpnet),
    "minilm_v2": load_model(filename_minilm),
    "mpnet_base_dot_v1": load_model(filename_mpnet),
    "ft_allmpnet_v2": load_model(filename_ft_allmpnet),
    "ft_minilm_v2": load_model(filename_ft_minilm),
    "ft_mpnet_base_dot_v1": load_model(filename_ft_mpnet),
}

# Function to save feedback to a file
def save_feedback_to_csv(model_choice, query, feedback, filename="feedback.csv"):
    relevant_ids = [key.split('_')[1] for key, value in feedback.items() if value == 'Relevant']
    not_relevant_ids = [key.split('_')[1] for key, value in feedback.items() if value == 'Not Relevant']

    try:
        with open(filename, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([model_choice, query, relevant_ids, not_relevant_ids])
        st.success("Feedback has been saved to CSV!")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Define the app layout and functionality
def main():
    st.markdown(
        """
        <style>
            body {
                background-color: #202124;
                color: #fff;
            }
            .main h1 {
                font-size: 64px;
                font-weight: 400;
                color: #4285f4;
                margin-bottom: 20px;
            }
            .search-container {
                text-align: center;
                margin-top: 20px;
            }
            .search-container input[type="text"], .search-container select {
                width: 400px;
                padding: 10px;
                font-size: 18px;
                border: none;
                border-radius: 24px;
                margin: 10px 0;
            }
            .search-container button {
                padding: 10px 20px;
                font-size: 18px;
                margin-top: 10px;
                border: none;
                border-radius: 24px;
                background-color: #4285f4;
                color: white;
                cursor: pointer;
            }
            .search-container button:hover {
                background-color: #357ae8;
            }
            .result-item {
                display: flex;
                border: 1px solid #ddd;
                margin: 10px auto;
                padding: 10px;
                border-radius: 5px;
                background-color: #303134;
                width: 80%;
            }
            .result-content {
                flex: 1;
            }
            .result-item h3 {
                margin: 0;
                font-size: 1.2em;
            }
            .result-item p {
                margin: 5px 0;
            }
            .pagination {
                text-align: center;
                margin: 20px 0;
            }
            .pagination a {
                margin: 0 5px;
                padding: 5px 10px;
                border: 1px solid #ddd;
                text-decoration: none;
                color: #fff;
                background-color: #333;
            }
            .pagination a.active {
                background-color: #4285f4;
            }
            .feedback {
                display: flex;
                justify-content: space-between;
            }
            .submit-button-container {
                text-align: center;
                margin-top: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("EduLearning")

    search_text = st.text_input("Search...", "")
    model_choice = st.selectbox("Select Model", ["allmpnet_v2", "minilm_v2", "mpnet_base_dot_v1", "ft_allmpnet_v2", "ft_minilm_v2", "ft_mpnet_base_dot_v1"])

    if st.button("Search"):
        if search_text and model_choice:
            model_key = model_choice.lower().replace(" ", "")
            model = model_dict.get(model_key)
            if model:
                if model_choice.startswith("ft_"):
                    results, _ = model.semantic_search_unstemmed(search_text, top_n=20, similarity_threshold=0.07, n_similar=5)
                else:
                    results, _ = model.semantic_search_unstemmed(search_text, top_n=20, similarity_threshold=0.07)
                st.session_state['results'] = results['results'][:20]  # Save top 20 results to session state
                st.session_state['search_text'] = search_text
                st.session_state['model_choice'] = model_choice

    if 'results' in st.session_state:
        display_results(st.session_state['results'], st.session_state['search_text'], st.session_state['model_choice'])

def display_results(results, search_text, model_choice):
    st.header("Search Results")
    
    if results:
        feedback = {}
        for index, result in enumerate(results):  # Display only the top 20 results
            st.write(f"Result {index}")
            title = result.get('item_translated_title', 'No title available')
            headline = result.get('item_translated_headline', 'No headline available')
            objectives = result.get('item_translated_objectives', 'No objectives available')
            feedback_key = f"feedback_{index}"
            
            st.markdown(
                f"""
                <div class="result-item">
                    <div class="result-content">
                        <h3>{title}</h3>
                        <p>{headline}</p>
                        <p><strong>Objectives:</strong> {objectives}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            # Use radio buttons directly with session state, including "No Selection"
            feedback_value = st.radio("Select feedback:", ["No Selection", "Relevant", "Not Relevant"], key=feedback_key)
            feedback[feedback_key] = feedback_value
        
        if st.button("Submit Feedback"):
            # Filter out "No Selection" entries
            feedback = {key: value for key, value in feedback.items() if value != "No Selection"}
            save_feedback_to_csv(model_choice, search_text, feedback)  # Save the feedback to a CSV file
            # st.success("Feedback has been saved to CSV!")
            # Reset feedback state
            for key in feedback:
                del st.session_state[key]  # Reset for a new session

if __name__ == "__main__":
    main()
