from src.experiments.amount_of_features import run as run_amount_of_features
from src.experiments.rotation import run as run_rotation
from src.experiments.scaling import run as run_scaling
from src.experiments.noise import run_salt_and_pepper, run_gaussian_noise
import matplotlib.pyplot as plt

run_amount_of_features()
run_rotation()
run_scaling()
run_salt_and_pepper()
run_gaussian_noise()

plt.show()
