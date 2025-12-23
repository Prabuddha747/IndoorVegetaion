# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- `NPK_New Dataset.xlsx` file in the `data/` directory

## Setup Instructions

### Option 1: Using the Setup Script (Recommended)

```bash
# Make the setup script executable (if needed)
chmod +x setup.sh

# Run the setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Running the Project

### Step 1: Prepare Your Dataset

Place your `NPK_New Dataset.xlsx` file in the `data/` directory:

```bash
mkdir -p data
# Copy your NPK_New Dataset.xlsx to data/
```

### Step 2: Train Models

Train all three models (run these in order):

```bash
# Train pH prediction model
python pH.py

# Train NHI estimation model
python NHI.py

# Train growth stage classification model
python plantgrowthstageclassification.py
```

**Note**: Training may take several minutes depending on your hardware. Each script will:
- Load and preprocess the data
- Train TabNet, LSTM, and TCN models
- Generate performance metrics
- Create visualizations
- Save trained models

### Step 3: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Troubleshooting

### Import Errors

If you see import errors, make sure:
1. Virtual environment is activated
2. All packages are installed: `pip install -r requirements.txt`

### Dataset Not Found

Ensure `NPK_New Dataset.xlsx` is in the `data/` directory:
```bash
ls data/NPK_New\ Dataset.xlsx
```

### Model Files Not Found

If the dashboard shows "models not loaded", run the training scripts first:
```bash
python pH.py
python NHI.py
python plantgrowthstageclassification.py
```

### CUDA/GPU Issues

The code automatically uses CPU if CUDA is not available. For GPU acceleration:
- Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Ensure CUDA drivers are installed

## Project Structure After Setup

```
research_paper/
├── data/
│   └── NPK_New Dataset.xlsx
├── models/                    # Created after training
│   ├── ph_tabnet/
│   ├── nhi_tabnet/
│   ├── growth_stage_tabnet/
│   └── *.pkl files
├── results/                   # Created after training
│   ├── plots/
│   └── metrics/
├── venv/                      # Virtual environment
├── pH.py
├── NHI.py
├── plantgrowthstageclassification.py
├── app.py
├── requirements.txt
└── README.md
```

## Next Steps

1. **Explore the Dashboard**: Navigate through different pages to see predictions and insights
2. **Review Visualizations**: Check `results/plots/` for generated visualizations
3. **Analyze Metrics**: Review `results/metrics/` for model performance data
4. **Customize**: Modify hyperparameters in the Python scripts for your specific use case

## Getting Help

- Check `README.md` for detailed documentation
- Review error messages for specific issues
- Ensure all dependencies are correctly installed

---

**Happy Cultivating! 🌱**

