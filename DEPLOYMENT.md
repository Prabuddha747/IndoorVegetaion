# Streamlit Cloud Deployment Guide

This guide will help you deploy the Precision Indoor Cultivation Dashboard to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your code pushed to GitHub repository: `https://github.com/Prabuddha747/IndoorVegetaion.git`
3. A Streamlit Cloud account (free tier available at https://streamlit.io/cloud)

## Deployment Steps

### Step 1: Push Code to GitHub

If you haven't already, push your code to GitHub:

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Sign in to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository: `Prabuddha747/IndoorVegetaion`
   - Select branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

### Step 3: Configure App Settings

Streamlit Cloud will automatically:
- Detect `requirements.txt` and install dependencies
- Use Python 3.8+ (default)
- Set up the app with your configuration

### Step 4: Access Your Deployed App

Once deployed, Streamlit Cloud will provide you with a URL like:
```
https://your-app-name.streamlit.app
```

## Important Notes

### Model Files

The app requires trained model files in the `models/` directory. Make sure:
- All model files are committed to GitHub (`.pkl`, `.pt`, `.zip` files)
- Models are not in `.gitignore` (they're currently tracked)

### Dataset

The dataset file (`data/NPK_New Dataset.xlsx`) should be:
- Committed to GitHub if it's small (< 100MB)
- Or uploaded to a cloud storage service and accessed via URL if larger

### Environment Variables

If you need to set environment variables:
1. Go to your app settings in Streamlit Cloud
2. Click "Advanced settings"
3. Add environment variables as needed

## Troubleshooting

### App Fails to Deploy

- Check that `requirements.txt` includes all dependencies
- Verify `app.py` is the correct entry point
- Check Streamlit Cloud logs for error messages

### Models Not Found

- Ensure model files are committed to GitHub
- Check file paths in `app.py` match the repository structure
- Verify models are not in `.gitignore`

### Import Errors

- Check `requirements.txt` has correct package versions
- Verify all Python files are in the repository
- Check Streamlit Cloud logs for specific import errors

## Updating Your App

To update your deployed app:

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update description"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy your app

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Cloud Community](https://discuss.streamlit.io/c/streamlit-cloud/9)

---

**Note**: The free tier of Streamlit Cloud has some limitations. For production use, consider upgrading to a paid plan.

