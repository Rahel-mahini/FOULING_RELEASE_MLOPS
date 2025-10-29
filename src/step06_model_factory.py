from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def build_model_from_config(model_config):
    """Convert a model config dictionary to an actual sklearn model instance."""
    model_type = model_config.get('type')
    
    if model_type == 'LinearRegression':
        return LinearRegression()
    elif model_type == 'Lasso':
        return Lasso(alpha=model_config.get('alpha', 1.0))
    elif model_type == 'Ridge':
        return Ridge(alpha=model_config.get('alpha', 1.0))
    elif model_type == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            random_state=model_config.get('random_state', 42)
        )
    elif model_type == 'DecisionTree':
        return DecisionTreeRegressor(
            max_depth=model_config.get('max_depth', None),
            random_state=model_config.get('random_state', 42)
        )
    elif model_type == 'SVR':
        return SVR(kernel=model_config.get('kernel', 'rbf'))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
