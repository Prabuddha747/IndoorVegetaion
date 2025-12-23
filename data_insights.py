"""
Dataset-Specific Insights Generator
===================================
Analyzes the dataset to generate data-driven insights and recommendations.
"""

import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NPK_New Dataset.xlsx")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

def analyze_dataset_insights(df):
    """
    Analyze dataset to generate data-driven insights and recommendations.
    
    Returns:
        dict: Dictionary containing insights for pH, NHI, and Growth Stage
    """
    insights = {
        'pH': {},
        'NHI': {},
        'growth_stage': {}
    }
    
    # Calculate NHI if not present
    if 'NHI' not in df.columns:
        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
            df['NHI'] = (
                0.4 * (df['nitrogen'] / df['nitrogen'].max()) +
                0.35 * (df['phosphorus'] / df['phosphorus'].max()) +
                0.25 * (df['potassium'] / df['potassium'].max())
            ) * 100
    
    # ===============================
    # pH Analysis
    # ===============================
    if 'pH' in df.columns:
        ph_data = df['pH'].dropna()
        insights['pH'] = {
            'mean': ph_data.mean(),
            'median': ph_data.median(),
            'std': ph_data.std(),
            'min': ph_data.min(),
            'max': ph_data.max(),
            'optimal_range_pct': (ph_data.between(6.0, 7.5).sum() / len(ph_data) * 100),
            'acidic_pct': (ph_data < 6.0).sum() / len(ph_data) * 100,
            'alkaline_pct': (ph_data > 7.5).sum() / len(ph_data) * 100,
            'skewness': ph_data.skew(),
            'recommendations': []
        }
        
        # Dataset-specific pH recommendations
        ph_mean = insights['pH']['mean']
        ph_std = insights['pH']['std']
        
        if ph_mean < 6.0:
            insights['pH']['recommendations'].append(
                f"⚠️ **Dataset Analysis**: {insights['pH']['acidic_pct']:.1f}% of samples are acidic (pH < 6.0). "
                f"Your dataset shows mean pH of {ph_mean:.2f}, indicating systematic acidity issues."
            )
            insights['pH']['recommendations'].append(
                f"📊 **Pattern Found**: pH ranges from {ph_data.min():.2f} to {ph_data.max():.2f}. "
                f"Consider bulk lime application across {insights['pH']['acidic_pct']:.0f}% of affected areas."
            )
        elif ph_mean > 7.5:
            insights['pH']['recommendations'].append(
                f"⚠️ **Dataset Analysis**: {insights['pH']['alkaline_pct']:.1f}% of samples are alkaline (pH > 7.5). "
                f"Your dataset shows mean pH of {ph_mean:.2f}, indicating systematic alkalinity."
            )
            insights['pH']['recommendations'].append(
                f"📊 **Pattern Found**: Consider acidifying amendments for {insights['pH']['alkaline_pct']:.0f}% of samples."
            )
        else:
            insights['pH']['recommendations'].append(
                f"✅ **Dataset Analysis**: {insights['pH']['optimal_range_pct']:.1f}% of samples are in optimal range (6.0-7.5). "
                f"Mean pH of {ph_mean:.2f} indicates generally healthy soil conditions."
            )
        
        # Variability insights
        if ph_std > 1.0:
            insights['pH']['recommendations'].append(
                f"📈 **High Variability Detected**: pH standard deviation is {ph_std:.2f}, indicating significant spatial/temporal variation. "
                f"Consider site-specific management rather than uniform application."
            )
        
        # Correlation insights
        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature']):
            ph_corr = df[['pH', 'nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature']].corr()['pH']
            strongest_corr = ph_corr.drop('pH').abs().idxmax()
            corr_value = ph_corr[strongest_corr]
            
            insights['pH']['recommendations'].append(
                f"🔗 **Correlation Insight**: pH shows strongest correlation with {strongest_corr} (r={corr_value:.3f}). "
                f"Monitoring {strongest_corr} can help predict pH changes."
            )
    
    # ===============================
    # NHI Analysis
    # ===============================
    if 'NHI' in df.columns:
        nhi_data = df['NHI'].dropna()
        insights['NHI'] = {
            'mean': nhi_data.mean(),
            'median': nhi_data.median(),
            'std': nhi_data.std(),
            'min': nhi_data.min(),
            'max': nhi_data.max(),
            'optimal_pct': (nhi_data >= 80).sum() / len(nhi_data) * 100,
            'good_pct': ((nhi_data >= 60) & (nhi_data < 80)).sum() / len(nhi_data) * 100,
            'warning_pct': ((nhi_data >= 30) & (nhi_data < 60)).sum() / len(nhi_data) * 100,
            'critical_pct': (nhi_data < 30).sum() / len(nhi_data) * 100,
            'recommendations': []
        }
        
        nhi_mean = insights['NHI']['mean']
        
        if nhi_mean < 30:
            insights['NHI']['recommendations'].append(
                f"🔴 **Dataset Analysis**: {insights['NHI']['critical_pct']:.1f}% of samples are in critical range (NHI < 30). "
                f"Mean NHI of {nhi_mean:.2f} indicates widespread nutrient deficiencies requiring immediate intervention."
            )
        elif nhi_mean < 60:
            insights['NHI']['recommendations'].append(
                f"🟡 **Dataset Analysis**: {insights['NHI']['warning_pct']:.1f}% of samples are below optimal (NHI 30-60). "
                f"Mean NHI of {nhi_mean:.2f} suggests systematic nutrient management improvements needed."
            )
        elif nhi_mean < 80:
            insights['NHI']['recommendations'].append(
                f"🟢 **Dataset Analysis**: {insights['NHI']['good_pct']:.1f}% of samples are in good range (NHI 60-80). "
                f"Mean NHI of {nhi_mean:.2f} indicates adequate nutrient levels with room for optimization."
            )
        else:
            insights['NHI']['recommendations'].append(
                f"✅ **Dataset Analysis**: {insights['NHI']['optimal_pct']:.1f}% of samples are optimal (NHI ≥ 80). "
                f"Mean NHI of {nhi_mean:.2f} indicates excellent nutrient health across the dataset."
            )
        
        # NPK balance analysis
        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
            npk_ratio = {
                'N': df['nitrogen'].mean(),
                'P': df['phosphorus'].mean(),
                'K': df['potassium'].mean()
            }
            total_npk = sum(npk_ratio.values())
            npk_pct = {k: (v/total_npk*100) for k, v in npk_ratio.items()}
            
            # Ideal ratio is approximately 4:2:3 (N:P:K) or 40%:20%:30%
            if npk_pct['N'] < 35:
                insights['NHI']['recommendations'].append(
                    f"📊 **NPK Imbalance**: Nitrogen represents {npk_pct['N']:.1f}% of total NPK (below ideal ~40%). "
                    f"Consider increasing N applications."
                )
            if npk_pct['P'] < 15:
                insights['NHI']['recommendations'].append(
                    f"📊 **NPK Imbalance**: Phosphorus represents {npk_pct['P']:.1f}% of total NPK (below ideal ~20%). "
                    f"Consider P supplementation."
                )
            if npk_pct['K'] < 25:
                insights['NHI']['recommendations'].append(
                    f"📊 **NPK Imbalance**: Potassium represents {npk_pct['K']:.1f}% of total NPK (below ideal ~30%). "
                    f"Consider K supplementation."
                )
    
    # ===============================
    # Growth Stage Analysis
    # ===============================
    growth_col = None
    for col in df.columns:
        if 'growth' in col.lower() or 'stage' in col.lower():
            growth_col = col
            break
    
    if growth_col:
        stage_counts = df[growth_col].value_counts()
        total = len(df)
        
        insights['growth_stage'] = {
            'distribution': stage_counts.to_dict(),
            'distribution_pct': {k: (v/total*100) for k, v in stage_counts.items()},
            'recommendations': []
        }
        
        # Stage-specific nutrient analysis
        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
            stage_nutrients = df.groupby(growth_col)[['nitrogen', 'phosphorus', 'potassium']].mean()
            
            for stage in stage_counts.index:
                pct = insights['growth_stage']['distribution_pct'][stage]
                nutrients = stage_nutrients.loc[stage]
                
                insights['growth_stage']['recommendations'].append(
                    f"📊 **{stage} Stage** ({pct:.1f}% of dataset): "
                    f"Avg N={nutrients['nitrogen']:.1f}, P={nutrients['phosphorus']:.1f}, K={nutrients['potassium']:.1f}. "
                    f"Dataset shows {stage_counts[stage]} samples in this stage."
                )
        
        # Dominant stage insight
        dominant_stage = stage_counts.idxmax()
        dominant_pct = insights['growth_stage']['distribution_pct'][dominant_stage]
        insights['growth_stage']['recommendations'].append(
            f"🌱 **Dataset Pattern**: {dominant_stage} is the dominant stage ({dominant_pct:.1f}% of samples). "
            f"Focus nutrient management strategies on this stage's requirements."
        )
    
    # Save insights to file
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, 'dataset_insights.json'), 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    return insights

def generate_dataset_recommendations(pred_value, metric_type, df, insights):
    """
    Generate dataset-specific recommendations based on prediction and dataset patterns.
    
    Args:
        pred_value: Predicted value (pH, NHI, or growth stage)
        metric_type: 'pH', 'NHI', or 'growth_stage'
        df: Dataset dataframe
        insights: Pre-computed insights dictionary
    
    Returns:
        list: List of dataset-specific recommendation strings
    """
    recommendations = []
    
    if metric_type == 'pH':
        if metric_type in insights:
            ph_insights = insights[metric_type]
            
            # Compare prediction to dataset distribution
            if pred_value < ph_insights['mean'] - ph_insights['std']:
                recommendations.append(
                    f"📉 **Dataset Context**: Your predicted pH ({pred_value:.2f}) is below dataset mean "
                    f"({ph_insights['mean']:.2f}). This is in the lower {((df['pH'] < pred_value).sum() / len(df) * 100):.1f}% "
                    f"of your dataset."
                )
            elif pred_value > ph_insights['mean'] + ph_insights['std']:
                recommendations.append(
                    f"📈 **Dataset Context**: Your predicted pH ({pred_value:.2f}) is above dataset mean "
                    f"({ph_insights['mean']:.2f}). This is in the upper {((df['pH'] > pred_value).sum() / len(df) * 100):.1f}% "
                    f"of your dataset."
                )
            
            # Compare to optimal range
            optimal_pct = ph_insights['optimal_range_pct']
            if 6.0 <= pred_value <= 7.5:
                recommendations.append(
                    f"✅ **Dataset Benchmark**: Your pH is in optimal range. "
                    f"{optimal_pct:.1f}% of samples in your dataset fall in this range."
                )
    
    elif metric_type == 'NHI':
        if metric_type in insights:
            nhi_insights = insights[metric_type]
            
            # Compare to dataset distribution
            if pred_value < nhi_insights['mean'] - nhi_insights['std']:
                recommendations.append(
                    f"📉 **Dataset Context**: Your predicted NHI ({pred_value:.2f}) is below dataset mean "
                    f"({nhi_insights['mean']:.2f}). This is in the lower {((df['NHI'] < pred_value).sum() / len(df) * 100):.1f}% "
                    f"of your dataset."
                )
            elif pred_value > nhi_insights['mean'] + nhi_insights['std']:
                recommendations.append(
                    f"📈 **Dataset Context**: Your predicted NHI ({pred_value:.2f}) is above dataset mean "
                    f"({nhi_insights['mean']:.2f}). This is in the upper {((df['NHI'] > pred_value).sum() / len(df) * 100):.1f}% "
                    f"of your dataset."
                )
            
            # Compare to dataset percentiles
            if pred_value >= 80:
                recommendations.append(
                    f"✅ **Dataset Benchmark**: Your NHI is in optimal range. "
                    f"{nhi_insights['optimal_pct']:.1f}% of samples in your dataset achieve this level."
                )
            elif pred_value < 30:
                recommendations.append(
                    f"🔴 **Dataset Benchmark**: Your NHI is critical. "
                    f"{nhi_insights['critical_pct']:.1f}% of samples in your dataset are in this range, "
                    f"indicating this is a common issue requiring attention."
                )
    
    elif metric_type == 'growth_stage':
        if metric_type in insights:
            stage_insights = insights[metric_type]
            if pred_value in stage_insights['distribution_pct']:
                pct = stage_insights['distribution_pct'][pred_value]
                count = stage_insights['distribution'][pred_value]
                recommendations.append(
                    f"📊 **Dataset Context**: {pred_value} represents {pct:.1f}% of your dataset "
                    f"({count} samples). This is a {'common' if pct > 30 else 'less common' if pct < 15 else 'moderate'} "
                    f"stage in your cultivation environment."
                )
    
    return recommendations

