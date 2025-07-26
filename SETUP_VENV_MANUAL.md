# Manual Setup Instructions for Virtual Environment

## Step 1: Create Virtual Environment
```bash
python3 -m venv stllms_venv
```

## Step 2: Activate Virtual Environment
```bash
source stllms_venv/bin/activate
```

## Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

## Step 4: Install PyTorch (Mac-specific)

### For Apple Silicon Mac (M1/M2/M3):
```bash
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
```

### For Intel Mac:
```bash
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

## Step 5: Install Other Dependencies
```bash
pip install -r requirements_mac.txt
```

## Step 6: Install Package in Development Mode
```bash
pip install -e .
```

## Usage

### To activate environment in future sessions:
```bash
source stllms_venv/bin/activate
```

### To deactivate environment:
```bash
deactivate
```

### To run the project:
```bash
# Generate responses
cd train-steering-vectors
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 10 --batch_size 2

# Train steering vectors
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 10 --batch_size 1
```

## Troubleshooting

### If you get "command not found: bc" error:
```bash
brew install bc
```

### If you get permission errors:
```bash
chmod +x setup_venv_mac.sh
```

### To check your Python version:
```bash
python3 --version
```

### To list installed packages:
```bash
pip list
```
