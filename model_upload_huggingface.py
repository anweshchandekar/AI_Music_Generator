from huggingface_hub import HfApi, HfFolder, Repository

# Paste your Hugging Face token here
token = "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"

# Login to Hugging Face
HfFolder.save_token(token)
print("Login successful")

# Define your repository name and local model path
repo_name = "AI_Music_Generator"
model_path = r"C:\Users\anwes\OneDrive\Desktop\AI_Music_Generator\scripts\AI_Music_Generator.h5"

# Set up the repository on Hugging Face
api = HfApi()
username = api.whoami()['name']
repo_id = f"{username}/{repo_name}"
repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

# Clone the repository to a local directory
repo_local_path = f"./{repo_name}"
repo = Repository(repo_local_path, clone_from=repo_url)

# Copy the model to the local repository directory
import shutil
shutil.copy(model_path, repo_local_path)

# Push the model to Hugging Face
repo.push_to_hub(commit_message="Initial model upload")
print("Model uploaded successfully!")