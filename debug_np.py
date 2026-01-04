import joblib
import pandas as pd
import numpy as np

# Load artifact
art = joblib.load('models/ELSA_neuralprophet_meta.joblib')

print("="*70)
print("NEURALPROPHET ARTIFACT DEBUG")
print("="*70)
print(f"Keys: {list(art.keys())}")
print(f"Model type: {type(art.get('neuralprophet'))}")
print(f"Model: {art.get('neuralprophet')}")
print(f"Has scaler: {art.get('scaler') is not None}")
print(f"Feature columns: {len(art.get('feature_columns', []))} features")
print(f"Features: {art.get('feature_columns')}")
print(f"Metrics: {art.get('metrics')}")

# Check if model has expected attributes
model = art.get('neuralprophet')
if model is not None:
    print(f"\nModel attributes:")
    print(f"  - has predict: {hasattr(model, 'predict')}")
    print(f"  - has make_future_dataframe: {hasattr(model, 'make_future_dataframe')}")
    
    # Try to see model config
    try:
        if hasattr(model, 'config_train'):
            print(f"  - config_train exists")
        if hasattr(model, 'config'):
            print(f"  - config exists")
    except Exception as e:
        print(f"  - Error accessing config: {e}")

print("="*70)
