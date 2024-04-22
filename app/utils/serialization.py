import pickle

def load_pickle(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def save_pickle(file, data):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)