import numpy as np

import global_var

p = 0.75

def aggregate_and_predict_proba(estimators, x):
    y_pred_list = []
    for estimator in estimators:
        y_pred_list.append(estimator.predict_proba(x))
    y_pred_array = np.stack(y_pred_list, axis=2)
    return y_pred_array

def aggregate_and_evaluate(estimators, x, y):
    classes = estimators[0].classes_
    y_pred_array = aggregate_and_predict_proba(estimators, x)
    y_pred_proba = np.mean(y_pred_array, axis=2)
    y_pred = classes.take(np.argmax(y_pred_proba, axis=1), axis=0)
    error_vector = y_pred != y
    error_rate = np.mean(error_vector, axis=0)
    return error_rate, error_vector

def arc(n_estimators, x, y, BaseClassifier):
    estimators = []
    n_samples = len(y)
    BITE_RATIO = global_var.get_bag_scale()
    bite_size = int(n_samples*BITE_RATIO)
    resample_index = np.random.choice(np.arange(n_samples), size=bite_size,
                                      replace=True) # Initial bite is bagging
    ob_error_history = np.zeros(n_estimators)
    for i_bite in range(n_estimators):
        x_resampled = x[resample_index]
        y_resampled = y[resample_index]
        estimator = BaseClassifier()
        estimators.append(estimator)
        estimator.fit(x_resampled, y_resampled)

        ob_index = [i for i in range(n_samples) if i not in resample_index] # index of out-of-bag instances
        error_rate, error_vector = aggregate_and_evaluate(estimators, x, y)
        ob_error = np.mean(error_vector[ob_index], axis=0) # out-of-bag error
        if i_bite > 0:
            ob_error_history[i_bite] = (1-p) * ob_error + p * ob_error_history[i_bite - 1] # smoothing estimated error
        else:
            ob_error_history[i_bite] = ob_error

        if i_bite == n_estimators - 1:
            # skip the last resampling
            break
        # resample on the entire training set
        threshold = ob_error_history[i_bite] / (1-ob_error_history[i_bite])
        resample_index = []

        while len(resample_index) < bite_size:
            index = np.random.randint(0, n_samples)
            # Alternative: sample on ob_index
            # index = np.random.choice(ob_index)
            if error_vector[index] or np.random.random() < threshold:
                resample_index.append(index)

    return estimators, ob_error_history

def bagging(n_estimators, x, y, BaseClassifier):
    estimators = []
    n_samples = len(y)
    BAG_RATIO = global_var.get_bag_scale()
    bag_size = int(n_samples*BAG_RATIO)
    
    for i_bag in range(n_estimators):
        resample_index = np.random.choice(np.arange(n_samples), size=bag_size,
                                      replace=True)
        x_resampled = x[resample_index]
        y_resampled = y[resample_index]
        estimator = BaseClassifier()
        estimators.append(estimator)
        estimator.fit(x_resampled, y_resampled)

    return estimators, None
