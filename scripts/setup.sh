#bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt 

# Clone the G4L repository
git clone https://github.com/gpt4free/gpt4local

# Navigate to the G4L repository
cd gpt4local

# Install the required packages
pip install -r requirements.txt

# Create a bash function that takes as parameter the path to the model and the url to download it and downloads the model if it does not exist
download_model() {
    MODEL_PATH=$1
    MODEL_URL=$2
    if [ ! -f $MODEL_PATH ]; then
        echo "Downloading $MODEL_URL to $MODEL_PATH from the current directory $PWD"
        curl -L $MODEL_URL > $MODEL_PATH
    fi
}

# Download the models
MODELS_FOLDER_PATH="./models"
MODEL_URLS=(
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q5_K_S.gguf"
    "https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf"
)
MODEL_NAMES=(
    "mistral-7b-instruct-v0.2.Q5_K_S.gguf"
    "orca-mini-3b-gguf2-q4_0.gguf"
)

for i in ${!MODEL_URLS[@]}; do
    download_model "$MODELS_FOLDER_PATH/${MODEL_NAMES[$i]}" "${MODEL_URLS[$i]}"
done
