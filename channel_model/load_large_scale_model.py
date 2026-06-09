import numpy as np
import pandas as pd

def get_large_scale_df(config: dict) -> pd.DataFrame:
    """
    Returns the large-scale fading DataFrame for the scenario named in config.

    Reads config["online"]["channel_model"]["channel_scenario"] and delegates
    to load_large_scale_model().  This is the single call-site used by
    run_online.py so that no DataFrame construction logic leaks into that script.

    Args:
        config (dict): Full simulation configuration loaded from
                       simulation_configuration.yaml.

    Returns:
        pd.DataFrame: Fading parameters indexed by elevation angle (10° – 90°).
    """
    scenario = config["online"]["channel_model"]["channel_scenario"]
    return load_large_scale_model(scenario)["large_scale_fading"]

def load_large_scale_model(scenario):
    """
    Loads the parameters for large-scale fading losses from 3GPP TR 38.811.

    Args:
        scenario (str): The deployment scenario. Can be 'dense_urban',
                        'urban', or 'sub_urban'.

    Returns:
        dict: A dictionary containing the 'large_scale' key, which holds a
              pandas DataFrame with the model parameters. The DataFrame has
              the following columns:
              - 'elevation': Elevation angles [deg]
              - 'los_prob': Line-of-sight probability [%]
              - 'sigma_los': Shadow fading std for LOS [dB]
              - 'sigma_nlos': Shadow fading std for NLOS [dB]
              - 'cl': Clutter loss for NLOS [dB]
              
    Requires:
        - numpy
        - pandas
    """
    elevation = np.arange(10, 91, 10)
    scenario = scenario.lower()

    if scenario == 'dense_urban':
        los_prob = np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1])
        sigma_los = np.array([2.9, 2.4, 2.7, 2.4, 2.4, 2.7, 2.6, 2.8, 0.6])
        sigma_nlos = np.array([17.1, 17.1, 15.6, 14.6, 14.2, 12.6, 12.1, 12.3, 12.3])
        cl = np.array([44.3, 39.9, 37.5, 35.8, 34.6, 33.8, 33.3, 33.0, 32.9])

    elif scenario == 'urban':
        los_prob = np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2])
        sigma_los = np.full(9, 4.0)
        sigma_nlos = np.full(9, 6.0)
        cl = np.array([44.3, 39.9, 37.5, 35.8, 34.6, 33.8, 33.3, 33.0, 32.9])

    elif scenario == 'sub_urban':
        los_prob = np.array([78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8])
        sigma_los = np.array([1.9, 1.6, 1.9, 2.3, 2.7, 3.1, 3.0, 3.6, 0.4])
        sigma_nlos = np.array([10.7, 10.0, 11.2, 11.6, 11.8, 10.8, 10.8, 10.8, 10.8])
        cl = np.array([29.5, 24.6, 21.9, 20.0, 18.7, 17.8, 17.2, 16.9, 16.8])
        
    else:
        raise ValueError("Scenario must be 'dense_urban', 'urban', or 'sub_urban'")

    # Create a dictionary of the data
    data = {
        'elevation': elevation,
        'los_prob': los_prob,
        'sigma_los': sigma_los,
        'sigma_nlos': sigma_nlos,
        'cl': cl
    }
    
    # Create the pandas DataFrame
    large_scale_df = pd.DataFrame(data)
    const = {'large_scale_fading': large_scale_df}
    return const
