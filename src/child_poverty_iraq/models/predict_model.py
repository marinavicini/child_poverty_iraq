import child_poverty_iraq.utils.constants as c
import child_poverty_iraq.models.train_model as tm


def predict_model(model, model_details, data):
    pca = model_details["pca"]
    scaler = model_details["scaler"]
    pca_components = model_details["pca_components"]
    boxcox = model_details["boxcox"]
    lmbda = model_details["lmbda"]

    # PCA components
    if pca_components is not None:
        inpt = tm.transform_pca(data[c.mosaiks_features], pca, scaler)
    else:
        inpt = scaler.transform(data[c.mosaiks_features])

    predictions = model.predict(inpt)

    # Boxcox
    if boxcox:
        predictions = tm.reverse_boxcox(predictions, lmbda)

    return predictions
