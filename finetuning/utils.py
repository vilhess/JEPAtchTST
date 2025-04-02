import json

def get_dim(dataset_name):
    
    assert dataset_name in ["nyc_taxi", "ec2_request_latency_system_failure", "msl", "smap", "smd", "swat"], f"Invalid dataset name: {dataset_name}"
    if dataset_name == "nyc_taxi" or dataset_name == "ec2_request_latency_system_failure":
        return 1
    elif dataset_name == "msl": 
        return 55
    elif dataset_name == "smap":
        return 25
    elif dataset_name == "smd":
        return 38
    elif dataset_name == "swat":
        return 51
    
def get_loaders(dataset, config):
    av_datasets = ["nyc_taxi", "smd", "smap", "msl", "swat", "ec2_request_latency_system_failure"]
    assert dataset in av_datasets, f"Dataset ({dataset}) should be in {av_datasets}"
    dim = get_dim(dataset)

    if dataset in ["ec2_request_latency_system_failure", "nyc_taxi"]:
        from dataset.nab import get_loaders as get_nab_loaders
        loaders = [get_nab_loaders(window_size=config.ws-1, root_dir="data/nab", dataset=dataset, batch_size=config.batch_size)]

    elif dataset in ["smap", "msl"]:
        from dataset.nasa import get_loaders as get_nasa_loaders, smapfiles, mslfiles
        file = smapfiles if dataset == "smap" else mslfiles
        loaders = [get_nasa_loaders(window_size=config.ws-1, root_dir="data/nasa", dataset=dataset, filename=f, batch_size=config.batch_size) for f in file]

    elif dataset == "smd":
        from dataset.smd import get_loaders as get_smd_loaders, machines
        loaders = [get_smd_loaders(window_size=config.ws-1, root_dir="data/smd/processed", machine=m, batch_size=config.batch_size) for m in machines]

    elif dataset == "swat":
        from dataset.swat import get_loaders as get_swat_loaders
        loaders = [get_swat_loaders(window_size=config.ws-1, root_dir="data/swat", batch_size=config.batch_size)]
        
    return loaders, dim

def load_results(filename="aucs.json"):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def save_results(filename, dataset, score):
    results = load_results(filename)
    results[dataset] = score
    with open(filename, "w") as f:
        json.dump(results, f)