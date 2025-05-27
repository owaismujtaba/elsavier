from src.dataset.bids import create_bids_dataset
from src.analysis.registery import EEGEpochExtractor
from src.visualizations.p_100_component import P100Plotter
import config as config

import pdb


if __name__== '__main__':
    
    if config.CREATE_BIDS_DATASET:
        create_bids_dataset(dataset_details=config.filepaths[8:])

    if config.PLOT_P100_COMP:
        epochs_extractor = EEGEpochExtractor()
        
        # Variable values for visual stimulus
        trial_mode, trial_unit, experiment_mode = '', 'Words', 'Experiment' 
        trial_boundary, trial_type, modality= 'Start', 'Stimulus', 'Pictures'
        visual_epochs = epochs_extractor.create_epochs(
            trial_mode=trial_mode, trial_type=trial_type,
            trial_unit=trial_unit, experiment_mode=experiment_mode,
            trial_boundary=trial_boundary, modality=modality
        )
        visual_amplitudes = {}
        for key, epochs in visual_epochs.items():
            subject_id, session_id = key[0], key[1]
            name = f"{trial_mode}_{trial_unit}_{experiment_mode}_{trial_boundary}_{trial_type}"
            name = f'{name}_p_100_component_visual'
            plotter = P100Plotter(channels=['PO3', 'POz', 'PO4'])
            plotter.plot_p100(
                epochs=epochs, name=name, 
                subject_id=subject_id, session_id=session_id)
            visual_amplitudes[key] = plotter.get_p100_amplitude(epochs)
        
        rest_epochs = epochs_extractor.create_epochs(
            trial_mode=trial_mode, trial_type='Fixation',
            trial_unit=trial_unit, experiment_mode=experiment_mode,
            trial_boundary=trial_boundary, modality=modality,
            tmin=0.2, tmax=0.9
        )
        rest_amplitudes = {}
        for key, epochs in rest_epochs.items():
            subject_id, session_id = key[0], key[1]
            name = f"{trial_mode}_{trial_unit}_{experiment_mode}_{trial_boundary}_{trial_type}"
            name = f'{name}_p_100_component_rest'
            plotter = P100Plotter(channels=['PO3', 'POz', 'PO4'])
            plotter.plot_p100(
                epochs=epochs, name=name, 
                subject_id=subject_id, session_id=session_id)
            rest_amplitudes[key] = plotter.get_p100_amplitude(epochs)
        


        print(rest_amplitudes, visual_amplitudes)