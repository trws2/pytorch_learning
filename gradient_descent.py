import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger

# generate a set of synthetic data using linear regression with added Gaussian noises
np.random.seed(42)
true_w = 2.0
true_b = 1.0

num_examples = 1200
x = np.random.rand(num_examples, 1)
y = true_w * x + true_b
noise = 0.1 * np.random.randn(num_examples, 1)
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

# scale data to make it zero mean and uni-variant
scaler = StandardScaler(with_mean=True, with_std=True)
train_x = scaler.fit_transform(train_x)
val_x = scaler.fit_transform(val_x)
test_x = scaler.fit_transform(test_x)

logger.info("training x:")
logger.info(f"{train_x[:10]}")
logger.info(f"{train_y[:10]}")

logger.info("validation x:")
logger.info(f"{val_x[:10]}")
logger.info(f"{val_y[:10]}")

logger.info("testing x:")
logger.info(f"{test_x[:10]}")
logger.info(f"{test_y[:10]}")

b = 0.1
w = 0.1
lr = 0.01
print(f"Initial: true_b = {true_b}, true_w = {true_w}, b = {b}, w = {w}")

all_losses = []
for iter in range(10000):
    yhat = train_x * w + b
    error = yhat - train_y
    loss = (error**2).mean()
    all_losses.append(loss)
    
    if iter > 0 and iter % 10 == 0:
        if all_losses:
            loss_diff = abs(all_losses[-1] - all_losses[-10]) / max(all_losses[-1], all_losses[-10])
            if loss_diff < 0.00000000001:
                print(
                    "loss change is too little, stop the training: current loss = {}, loss before = {}, diff_% = {}".format(
                        all_losses[-1], all_losses[-10], loss_diff
                    )
                )
                break
    print(f"iteration {iter}: loss = {all_losses[-1]}")

    b_grad = 2 * error.mean()
    w_grad = 2 * (train_x * error).mean()

    b = b - lr * b_grad
    w = w - lr * w_grad

print(f"Final: true_b = {true_b}, true_w = {true_w}, b = {b}, w = {w}, loss = {loss}")

