import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import numpy as np

def compute_sdt_metrics_lib(df):
    """
    Oblicza parametry SDT uzywajac bibliotek zewnetrznych.
    Służy jako 'ground truth' do udowodnienia prawidłowości wzorów manualnych.
    Metoda norm.ppf to nic innego jak Percent Point Function - odwrotność dystrybuanty normalnej.
    """
    if 'klasa_wyniku' in df.columns:
        TP = (df['klasa_wyniku'] == 'TP').sum()
        FN = (df['klasa_wyniku'] == 'FN').sum()
        FP = (df['klasa_wyniku'] == 'FP').sum()
        TN = (df['klasa_wyniku'] == 'TN').sum()
    else:
        TP = ((df['bodziec_obecny'] == 1) & (df['odpowiedz'] == 1)).sum()
        FN = ((df['bodziec_obecny'] == 1) & (df['odpowiedz'] == 0)).sum()
        FP = ((df['bodziec_obecny'] == 0) & (df['odpowiedz'] == 1)).sum()
        TN = ((df['bodziec_obecny'] == 0) & (df['odpowiedz'] == 0)).sum()
        
    HR = (TP + 0.5) / (TP + FN + 1.0)
    FAR = (FP + 0.5) / (FP + TN + 1.0)
    
    # Scipy implementation of inverse normal CDF
    d_prime = norm.ppf(HR) - norm.ppf(FAR)
    c = -0.5 * (norm.ppf(HR) + norm.ppf(FAR))
    
    return {
        'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
        'HR': HR, 'FAR': FAR,
        'd_prime': float(d_prime), 'criterion_c': float(c)
    }

def fit_psychometric_curve_lib(levels, responses, threshold=0.75):
    """
    Dopasowanie krzywej za pomocą statsmodels - Generalized Linear Model.
    Rodzina: Binomial()
    """
    if len(levels) == 0:
        return None, None, None
        
    X = sm.add_constant(levels) # [1, x]
    model = sm.GLM(responses, X, family=sm.families.Binomial())
    try:
        res = model.fit()
        b0, b1 = res.params
        
        logit_p = np.log(threshold / (1 - threshold))
        thr = (logit_p - b0) / b1 if b1 != 0 else np.nan
        
        return float(b0), float(b1), float(thr)
    except Exception as e:
        # fall back if fitting fails
        return None, None, None
