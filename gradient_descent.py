
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger

# generate a set of synthetic data using linear regression with added Gaussian noises
np.random.seed(37)
w = 2.0
b = 1.0

num_examples = 12000
x = np.arange(num_examples)
y = w * x + b
noise = np.random.normal(0, 5, num_examples)
y = y + noise

# split them into train, val, and test
ids = np.arange(num_examples)
np.random.shuffle(ids)

num_examples_train = 9000
num_examples_val = 1000
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

logger.info("training x:")
logger.info(f"{train_x[:10]}")
logger.info(f"{train_y[:10]}")

logger.info("validation x:")
logger.info(f"{val_x[:10]}")
logger.info(f"{val_y[:10]}")

logger.info("testing x:")
logger.info(f"{test_x[:10]}")
logger.info(f"{test_y[:10]}")

