import numpy as np
from sklearn.preprocessing import StandardScaler
from loguru import logger
import matplotlib.pyplot as plt

# This script generate synthetic data using linear regression model with added Gaussian noises and 
# split into training, validation, and testing data. 
#
# It then trains the linear model with hyper-parameter tunning (learning rate) using validation data, 
# and then plot the data and model on training, validation and testing data.
# 
# Note that this approach does not treat validation data as a part of training data (e.g. k-fold cross validation)
# to tune the hyper parameter.
#
# Data scaled with zero-mean and uni-variance with training model parameters. Data is scaled back to
# original scale when doing plotting.


DEBUG = False
np.random.seed(42)
TRUE_W = 2.0
TRUE_B = 1.0

# generate a set of synthetic data using linear regression with added Gaussian noises
def generate_date(debug=False):
    num_examples = 1200
    # x = np.random.rand(num_examples, 1)
    x = np.arange(num_examples).reshape(-1, 1)
    y = TRUE_W * x + TRUE_B
    noise = np.random.normal(loc=0, scale=50.0, size=(num_examples, 1))
    y = y + noise

    # split them into train, val, and test
    ids = np.arange(num_examples)        
    np.random.shuffle(ids)

    num_examples_train = 900
    num_examples_val = 100
    # num_examples_test = num_examples - num_examples_train - num_examples_val

    training_ids = ids[:num_examples_train]
    ids = ids[num_examples_train:]
    val_ids = ids[:num_examples_val]
    ids = ids[num_examples_val:]
    test_ids = ids

    train_x = x[training_ids]
    train_y = y[training_ids]

    val_x = x[val_ids]
    val_y = y[val_ids]

    test_x = x[test_ids]
    test_y = y[test_ids]

    print(len(train_x), len(train_y))
    print(len(val_x), len(val_y))
    print(len(test_x), len(test_y))        

    # scale data to make it zero mean and uni-variant
    scaler_train_x = StandardScaler(with_mean=True, with_std=True)
    train_x = scaler_train_x.fit_transform(train_x)
    scaler_val_x = StandardScaler(with_mean=True, with_std=True)
    val_x = scaler_val_x.fit_transform(val_x)
    scaler_test_x = StandardScaler(with_mean=True, with_std=True)
    test_x = scaler_test_x.fit_transform(test_x)

    if DEBUG:
        logger.info("training x:")
        logger.info(f"{train_x[:10]}")
        logger.info(f"{train_y[:10]}")

        logger.info("validation x:")
        logger.info(f"{val_x[:10]}")
        logger.info(f"{val_y[:10]}")

        logger.info("testing x:")
        logger.info(f"{test_x[:10]}")
        logger.info(f"{test_y[:10]}")

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        scaler_train_x,
        scaler_val_x,
        scaler_test_x,
    )


# train a linear regression model with gradient descent using mean square loss function
def train(lr=0.1, debug=False):
    # initial parameter values
    b = 0.1
    w = 0.1
    if debug:
        logger.info(f"Initial: true_b = {TRUE_B}, true_w = {TRUE_W}, b = {b}, w = {w}")

    all_losses = []
    for iter in range(10000):
        yhat = train_x * w + b
        error = yhat - train_y
        loss = (error**2).mean()
        all_losses.append(loss)

        if iter > 0 and iter % 10 == 0:
            if all_losses:
                loss_diff = abs(all_losses[-1] - all_losses[-10]) / max(
                    all_losses[-1], all_losses[-10]
                )
                if loss_diff < 0.000000001:
                    if debug:
                        logger.info(
                            "loss change is too little at iteration {}, stop the training: current loss = {}, loss before = {}, diff_% = {}".format(
                                iter, all_losses[-1], all_losses[-10], loss_diff
                            )
                        )
                    break
                if debug:
                    logger.info(f"iteration {iter}: loss = {all_losses[-1]}")

        b_grad = 2 * error.mean()
        w_grad = 2 * (train_x * error).mean()

        b = b - lr * b_grad
        w = w - lr * w_grad

    if debug:
        logger.info(
            f"Result: true_b = {TRUE_B}, true_w = {TRUE_W}, b = {b}, w = {w}, loss = {loss}"
        )

    if debug:
        # Plot the loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(all_losses, label=f"Loss at learning rate = {lr}", color="blue")
        plt.title(f"Loss Curve During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.yscale("log")  # Optional: use logarithmic scale for better visibility
        plt.grid(True)
        plt.legend()
        plt.show()

    return w, b, loss


def eval(x, y, w, b, x_scalar=None, plot=False, title="", train_x=None, train_x_scalar=None):
    error = x * w + b - y
    loss = (error**2).mean()

    if plot:
        plt.figure(figsize=(10, 6))
        if x_scalar:
            plt.scatter(
                x_scalar.inverse_transform(x),
                y,
                color="blue",
                label="Data Points",
                alpha=0.5,
            )
        else:
            plt.scatter(x, y, color="blue", label="Data Points", alpha=0.5)

        x_line = np.linspace(train_x.min(), train_x.max(), 100).reshape(-1, 1)
        y_line = x_line * w + b

        if train_x_scalar:
            plt.plot(
                train_x_scalar.inverse_transform(x_line),
                y_line,
                color="red",
                label="Regression Line",
            )
        else:
            plt.plot(x_line, y_line, color="red", label="Regression Line")

        plt.title(f"{title} and Regression Line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    return loss


(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    scaler_train_x,
    scaler_val_x,
    scaler_test_x,
) = generate_date(debug=DEBUG)

all_lrs = [1, 0.1, 0.05, 0.01, 0.0001, 0.00001]
results = {}
for lr in all_lrs:
    w, b, train_loss = train(lr, debug=DEBUG)
    val_loss = eval(val_x, val_y, w, b, x_scalar=scaler_val_x)
    results[lr] = {
        "w": w,
        "b": b,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    logger.info(f"Validation loss at learning rate {lr} = {val_loss}")

min_val_loss = min([x["val_loss"] for x in results.values()])
final_parameters = [(x["w"], x["b"]) for x in results.values() if x["val_loss"] == min_val_loss]
final_w, final_b = final_parameters[0][0], final_parameters[0][1]

# final result
train_loss = eval(train_x, train_y, final_w, final_b, x_scalar=scaler_train_x, plot=True, title="Training Data", train_x=train_x, train_x_scalar=scaler_train_x)
val_loss = eval(val_x, val_y, final_w, final_b, x_scalar=scaler_val_x, plot=True, title="Validation Data", train_x=train_x, train_x_scalar=scaler_train_x)
test_loss = eval(test_x, test_y, final_w, final_b, x_scalar=scaler_test_x, plot=True, title="Testing Data", train_x=train_x, train_x_scalar=scaler_train_x)
