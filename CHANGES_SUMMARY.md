# COT-Steering Mac Compatibility & API Updates

## Summary of Changes Made

### 🖥️ **Mac Compatibility Changes**

#### 1. **Device Detection & Support**
- ✅ Updated `load_model_and_vectors()` to auto-detect device (CUDA/MPS/CPU)
- ✅ Added proper dtype handling (float32 for CPU/MPS, bfloat16 for CUDA)
- ✅ Updated all hardcoded CUDA references to use device detection
- ✅ Added conditional CUDA memory cleanup

#### 2. **Environment Setup**
- ✅ Created `requirements_mac.txt` for pip-based installation
- ✅ Created `environment_mac.yaml` for conda (removed Linux-specific packages)
- ✅ Created `setup_venv_mac.sh` for automated virtual environment setup
- ✅ Created `SETUP_VENV_MANUAL.md` for manual setup instructions

### 🤖 **API Provider Updates**

#### 3. **Removed OpenRouter Dependency**
- ✅ Eliminated OpenRouter API calls entirely
- ✅ Now uses OpenAI and Anthropic APIs directly
- ✅ Cleaner, more reliable API integration

#### 4. **Updated Default Models**
- ✅ **OpenAI**: Default changed to `gpt-4o-mini` (from gpt-4.1)
- ✅ **Anthropic**: Default changed to `claude-3.5-sonnet` (from claude-3-7-sonnet)
- ✅ Added support for all major OpenAI models (gpt-4o, gpt-4, gpt-3.5-turbo)
- ✅ Added support for all Claude models (3-opus, 3-sonnet, 3-haiku, 3.5-sonnet, 3.5-haiku)

#### 5. **Environment Variable Support**
- ✅ Added `ANNOTATION_MODEL` environment variable
- ✅ Updated `.env.example` with new configuration options
- ✅ Removed OpenRouter API key requirement

### 🧠 **Enhanced Functionality**

#### 6. **Emotional Reasoning Support**
- ✅ Added emotional reasoning categories:
  - `depressive-thinking`
  - `anxious-thinking` 
  - `negative-attribution`
  - `pessimistic-projection`
- ✅ Added `analyze_emotional_content()` function
- ✅ Added `generate_and_analyze_emotional()` function
- ✅ Updated steering configurations for all models

#### 7. **Improved Annotation System**
- ✅ Added `include_emotional` parameter to annotation functions
- ✅ Support for both cognitive and emotional annotation frameworks
- ✅ Configurable annotation model via environment variables

## 📁 **New Files Created**

1. `requirements_mac.txt` - Mac-specific pip requirements
2. `environment_mac.yaml` - Mac-specific conda environment
3. `setup_venv_mac.sh` - Automated virtual environment setup script
4. `SETUP_VENV_MANUAL.md` - Manual setup instructions
5. `.env.example` - Updated environment variables template
6. `CHANGES_SUMMARY.md` - This summary document

## 🚀 **How to Use**

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup_venv_mac.sh
./setup_venv_mac.sh
```

### Option 2: Manual Setup
Follow instructions in `SETUP_VENV_MANUAL.md`

### Configuration
1. Copy `.env.example` to `.env`
2. Add your API keys:
   - `OPENAI_API_KEY` for GPT models
   - `ANTHROPIC_API_KEY` for Claude models
3. Optionally set `ANNOTATION_MODEL` (defaults to `gpt-4o-mini`)

### Running the Project
```bash
source stllms_venv/bin/activate  # Activate environment

# Generate responses (uses DeepSeek models)
cd train-steering-vectors
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 10

# Train steering vectors (uses OpenAI/Anthropic for annotation)
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 10
```

## 🔧 **Technical Improvements**

- **Device Auto-Detection**: Works on Apple Silicon (M1/M2/M3), Intel Macs, and CUDA systems
- **Memory Management**: Proper cleanup for different device types
- **Error Handling**: Better error messages for unsupported models
- **Flexible Configuration**: Environment-based configuration for easy switching between models
- **Cost Optimization**: Uses cheaper models (gpt-4o-mini) by default while maintaining quality

## ✅ **Compatibility**

- ✅ **Apple Silicon Macs** (M1/M2/M3) with MPS support
- ✅ **Intel Macs** with CPU inference
- ✅ **Linux/Windows** with CUDA (original functionality preserved)
- ✅ **Virtual Environment** (venv) support
- ✅ **Conda Environment** support (with updated yaml)

All changes maintain backward compatibility while adding Mac support and improving the overall user experience.
