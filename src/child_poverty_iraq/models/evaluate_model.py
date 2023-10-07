import matplotlib.pyplot as plt
import seaborn as sns
import child_poverty_iraq.data.load_data as ld


def scatter_plot(y_train_pred, y_train, y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(6, 5))  # , dpi=200)
    ax.scatter(y_train, y_train_pred)
    plt.title("Train")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    ax.set_aspect("equal")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))  # , dpi=200)
    ax.scatter(y_test, y_test_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Test")
    ax.set_aspect("equal")
    plt.show()


def hist_errors(y_train_pred, y_train, y_test, y_test_pred):
    errors_train = y_train - y_train_pred
    errors_test = y_test - y_test_pred

    fig, ax = plt.subplots(figsize=(8, 5))  # , dpi=200)
    sns.histplot(errors_train, kde=True, ax=ax, color="#FF817E")
    plt.xlabel("Target Error")
    plt.title("Histogram of Error (true - predicted)")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))  # , dpi=200)
    sns.histplot(errors_test, kde=True, color="#FF817E")
    plt.xlabel("Target Error")
    plt.title("Histogram of Error (true - predicted)")
    plt.show()


def get_scale_legend(data, col_to_plot="target_error"):
    max_scale = abs(data[col_to_plot]).max()
    return max_scale


def map_errors(data, col_to_plot="target_error"):
    max_scale = get_scale_legend(data, col_to_plot="target_error")

    # Get country shapes
    geom_adm0 = ld.get_mosaiks_geom_adm0()

    # Create a plot using matplotlib
    fig, ax = plt.subplots(figsize=(13, 5))  # , dpi=200)

    # Plot world countries
    geom_adm0[geom_adm0["shapeName"] != "Antarctica"].plot(ax=ax, color="#b0afae")

    # Plot countries with national poverty estimates
    data.plot(
        ax=ax,
        column=col_to_plot,
        cmap="Spectral",
        legend=True,
        vmin=-max_scale,
        vmax=max_scale,
    )

    plt.title(f"Errors of Moderate Prevalence")
    plt.axis("off")
    plt.show()


def map_irq_error(data, col_to_plot="target_error", error_scale=None):
    if error_scale is None:
        error_scale = get_scale_legend(data, col_to_plot="target_error")

    # Create a plot using matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))  # , dpi=200)

    # Plot countries with ADM1 error poverty estimates
    data[data["countrycode"] == "IRQ"].plot(
        ax=ax,
        column=col_to_plot,
        cmap="Spectral",
        legend=True,
        vmin=-error_scale,
        vmax=error_scale,
    )

    plt.title("Iraq ADM1 Errors of Moderate Prevalence")
    plt.axis("off")
    plt.show()


def compare_irq_pred(data, target, cmap="YlOrRd"):
    # Predicted
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Select bounds for plots
    if target == "sumpoor_mod":
        vmin = 0
        vmax = 4
    else:
        vmin = 0
        vmax = 100

    data[data["countrycode"] == "IRQ"].plot(
        ax=ax[0], column=target, cmap=cmap, legend=True, vmin=vmin, vmax=vmax
    )
    ax[0].set_title("TRUE Iraq ADM1 Prevalence")
    ax[0].axis("off")

    data[data["countrycode"] == "IRQ"].plot(
        ax=ax[1], column="predictions", cmap=cmap, legend=True, vmin=vmin, vmax=vmax
    )
    ax[1].set_title("PRED Iraq ADM1 Prevalence")
    ax[1].axis("off")
    plt.show()


def map_irq_training(data):
    # Create a plot using matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))  # , dpi=200)

    # Plot countries with ADM1 error poverty estimates
    data[data["countrycode"] == "IRQ"].plot(
        ax=ax, column="is_training", cmap="PiYG", legend=True
    )

    plt.title("Iraq ADM1 Errors of Moderate Prevalence")
    plt.axis("off")
    plt.show()
