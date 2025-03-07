import pandas as pd
import numpy as np

def get_coefficient_df(model, x_columns):
    """
    Create a DataFrame with model coefficients and their corresponding feature names
    
    Parameters:
    model: Fitted model with coef_ attribute (like LogisticRegression, LinearSVC)
    x_columns: List of feature names used for training
    
    Returns:
    pandas.DataFrame: DataFrame with features and their coefficients
    """
    # Extract coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    else:
        raise AttributeError("Model doesn't have coef_ attribute")
    
    # Check length
    if len(coefs) != len(x_columns):
        raise ValueError(f"Length mismatch: {len(coefs)} coefficients but {len(x_columns)} features")
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'Feature': x_columns,
        'Coefficient': coefs
    })
    
    # Sort by absolute coefficient value (descending)
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    # Drop the absolute column after sorting
    coef_df = coef_df.drop('Abs_Coefficient', axis=1)
    
    return coef_df
