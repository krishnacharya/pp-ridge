from pathlib import Path

def project_root():
    cur_dir = Path(__file__).absolute().parent.parent.parent
    return cur_dir

def data_root():
    res = project_root() / "datasets"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_root():
    res = data_root() / "raw"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_root():
    res = data_root() / "processed"
    res.mkdir(parents=True, exist_ok=True)
    return res

def output_root():
    res = project_root() / "output"
    res.mkdir(parents=True, exist_ok=True)
    return res

def output_dataset_root(dataset:str):
    res = output_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res