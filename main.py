from src.dataset.bids import create_bids_dataset
from src.analysis.registery import Registery
import config as config




if __name__== '__main__':
    
    if config.CREATE_BIDS_DATASET:
        create_bids_dataset(dataset_details=config.filepaths[8:])

    if config.PLOT_P100_COMP:
        registery = Registery()
        registery.plot_occipital_all_subjects()

