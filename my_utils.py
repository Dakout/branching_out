import os
import dill
import h5py
import numpy as np


def get_necessary_dtype(k: int) -> np.dtype:
    """
    Returns for a given number the minimum necessary numpy
    dtype to be able to represent this number
    """
    if k > 64:
        raise ValueError("dtype > 64 is not supported -> too much memory.")
    return np.min_scalar_type(2**k - 1)


def save_dict_to_hdf5(group, dictionary):
    """
    Iteratively converts a dict to h5 format.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            # Create a subgroup for nested dictionary
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, item)
        else:
            # Save dataset for other items
            try:
                group[key] = item
            except:
                pass
    return


def save_general_file(
    store_dir: str,
    file_name: str,
    obj_to_store,
    file_type: str = "pkl",
    overwrite: bool = False,
) -> None:
    """
    Checks if folder and file exist and saves accordingly.
    """
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    if file_type == "h5":
        if "h5" not in file_name:
            file_name = file_name + ".h5"

    # do not overwrite existing files
    if os.path.isfile(os.path.join(store_dir, file_name)) and not overwrite:
        print(f"File {file_name} exists already!!")
    else:
        if (file_type == "h5") or (".h5" in file_name):
            with h5py.File(os.path.join(store_dir, file_name), "w") as f:
                if isinstance(obj_to_store, np.ndarray):
                    f.create_dataset("array", data=obj_to_store)
                elif isinstance(obj_to_store, list):
                    array_data = np.array(obj_to_store, dtype=object)
                    f.create_dataset(
                        "list",
                        data=array_data,
                        dtype=h5py.special_dtype(vlen=np.dtype("float64")),
                    )
                elif type(obj_to_store) == dict:
                    save_dict_to_hdf5(f, obj_to_store)
        else:
            with open(file=os.path.join(store_dir, file_name), mode="wb") as f:
                dill.dump(obj_to_store, file=f)
    return


def load_dict_from_hdf5(group):
    """
    Recursively load dictionary from HDF5.
    """
    result = {}
    # a numpy array is stored as a dict with a single key 'array'
    if ("array" in group.keys()) and (len(group.keys()) == 1):
        return group["array"][:]
    # a list is stored as a dict with a single key 'list'
    if ("list" in group.keys()) and (len(group.keys()) == 1):
        return list(group["list"][:])
    # traverse the stored dict
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = load_dict_from_hdf5(item)
        else:
            result[key] = item[()]
    return result


def load_file(filename: str, file_type: str = "pkl") -> dict:
    """
    Loads the file specified by the filename path.
    """
    if (file_type == "h5") or (".h5" in filename):
        if "h5" not in filename:
            filename = filename + ".h5"
        # Load the dictionary from the HDF5 file
        with h5py.File(filename, "r") as f:
            d = load_dict_from_hdf5(f)
    else:
        print("in dill", filename)
        with open(filename, "rb") as f:
            d = dill.load(f)
    return d
