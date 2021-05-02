import pandas as pd
import numpy as np
labels_csv = pd.read_csv("./dog-breed-identification/labels.csv")
labels = labels_csv["breed"]
labels = np.array(labels)
unique_breeds = np.unique(labels)
