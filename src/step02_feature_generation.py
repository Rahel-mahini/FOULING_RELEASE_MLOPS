# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:49:10 2025

@author: RASULEVLAB
"""

# step02-feature_generation.py
from combinatorixPy import initialize_dask_cluster, get_result
from dask.distributed import Client

def generate_combinatorial_descriptors(config):
    """
    Generate combinatorial descriptors using combinatorixPy with Dask.
    Returns path to generated dataset.
    """
    cluster = initialize_dask_cluster(config)
    client = Client(cluster)

    print(" Dask client initialized.")

    descriptors_file = config['data']['descriptors_file']
    fractions_file = config['data']['fractions_file']
    output_path = config['data']['output_path']

    
    threshold_const = config['combinatorix']['threshold_const']
    threshold_corr = config['combinatorix']['threshold_corr']
    batch_num = config['combinatorix']['batch_num']

    result_path = get_result(
        descriptors_file,
        fractions_file,
        output_path,
        threshold_const,
        threshold_corr,
        batch_num,
        client
    )

    print(f" Combinatorial descriptor dataset saved at: {result_path}")

    client.close()
    cluster.close()

    return result_path