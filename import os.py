import os
import pickle

filename = 'model.pkl'
if not os.path.exists(filename):
    print(f"Model file not found at {filename}")
else:
    clf = pickle.load(open(filename, 'rb'))
