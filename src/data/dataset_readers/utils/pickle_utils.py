import os
import pickle


def save_pkl(instances, pickle_dict, is_training):
    if is_pickle_dict_valid(pickle_dict):
        os.makedirs(pickle_dict["path"], exist_ok=True)
        with open(get_pkl_path(pickle_dict, is_training), "wb") as dataset_file:
            pickle.dump(instances, dataset_file)


def load_pkl(pickle_dict, is_training):
    try:
        with open(get_pkl_path(pickle_dict, is_training), "rb") as dataset_pkl:
            return pickle.load(dataset_pkl)
    except Exception as e:
        return None


def get_pkl_path(pickle_dict, is_training):
    return os.path.join(
        pickle_dict["path"],
        f"{pickle_dict['file_name']}_{'train' if is_training else 'dev'}.pkl",
    )


def is_pickle_dict_valid(pickle_dict):
    if pickle_dict is None:
        return False
    file_name = pickle_dict.get("file_name", None)
    path = pickle_dict.get("path", None)
    return file_name is not None and path is not None
