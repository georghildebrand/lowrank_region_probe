from sklearn.linear_model import LogisticRegression, Ridge
import numpy as np

def fit_local_models(selected_cells, X, y, model_type="logreg"):
    """
    Fits simple local models for each selected cell.
    """
    results = []
    X_np = X.cpu().numpy() if hasattr(X, "cpu") else X
    y_np = y.cpu().numpy().flatten() if hasattr(y, "cpu") else y.flatten()
    
    for cell in selected_cells:
        idx = cell["member_indices"]
        if len(idx) < 5: # Not enough to fit anything
            continue
            
        X_cell = X_np[idx]
        y_cell = y_np[idx]
        
        # Check if only one class present
        if len(np.unique(y_cell)) < 2 and model_type == "logreg":
            acc = 1.0 # Trivial
            model_info = {"type": "trivial", "coeff": None}
        else:
            if model_type == "logreg":
                model = LogisticRegression(penalty='l2', C=1.0)
                model.fit(X_cell, y_cell)
                acc = model.score(X_cell, y_cell)
                model_info = {"type": "logreg", "coeffs": model.coef_.tolist()}
            elif model_type == "ridge":
                model = Ridge(alpha=1.0)
                model.fit(X_cell, y_cell)
                acc = model.score(X_cell, y_cell) # R^2
                model_info = {"type": "ridge", "coeffs": model.coef_.tolist()}
            else:
                acc = 0.0
                model_info = {"type": "unknown"}
                
        results.append({
            "cell_index": cell["cell_index"],
            "local_accuracy": acc,
            "model_info": model_info
        })
        
    return results
