import numpy as np
import pandas as pd
from scipy.stats import qmc
import sys
import re
from pathlib import Path
from IPython.display import display

# perform Latin Hypercube Sampling for the parameters and store the results in a CSV file
def latin_hypercube_sampling(
    random_seed: int,
    name_prefix: str,
    property_file_names_path: str,
    output_file_path: str,
    n_samples: int,
    param_names: list,
    lower_bounds: list,
    upper_bounds: list,
    property_list: list,
    ref_block_top_depth: float,
    ref_block_bottom_depth: float,
    show_results: bool = True
    ):

    # Create the output directory if it doesn't exist
    output_file_path = Path(output_file_path)
    output_file_path.mkdir(parents=True, exist_ok=True)

    # Load property file names
    property_file_names = np.loadtxt(property_file_names_path, delimiter=",", dtype=str)

    # sort the file names by the number in the name
    def extract_number(filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else float('inf')

    # Collect sorted file-name lists per property into a dict
    prop_files = {}
    for prop in property_list:
        prop_files[prop] = sorted(
            [name for name in property_file_names if prop.upper() in name.upper()],
            key=extract_number
        )

    # Check that every property has file names
    for prop in property_list:
        if not prop_files[prop]:
            print(f"Error: {prop} file names not found.")
            sys.exit(1)

    # Check all properties have the same number of files
    counts = {prop: len(prop_files[prop]) for prop in property_list}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Mismatched number of files across properties: {counts}")

    num_pairs = next(iter(counts.values()))

    if n_samples > num_pairs:
        raise ValueError(
            f"Cannot sample {n_samples} unique property sets: only {num_pairs} available."
        )

    # Latin Hypercube Sampling for parameters
    sampler = qmc.LatinHypercube(d=len(param_names), seed=random_seed)
    sample = sampler.random(n=n_samples)
    sample_params = qmc.scale(sample, lower_bounds, upper_bounds)
    df_params = pd.DataFrame(sample_params, columns=param_names)

    # Store file names for each property, with prefix
    prefix = "data_properties/"
    for prop in property_list:
        df_params[f"{prop}_file"] = [f"{prefix}{prop_files[prop][i]}" for i in range(n_samples)]


    # Calculate stress state parameters
    df_params['beta'] = df_params['SH_azi_deg'] - 90  # Rotate from SH to x-axis
    df_params['cos_2beta'] = np.cos(np.radians(2 * df_params['beta']))
    df_params['sin_2beta'] = np.sin(np.radians(2 * df_params['beta']))

    # calculate the stress gradients in kPa/km
    df_params['sigma_x_grad'] = (df_params['SH_MPa/km'] + df_params['Sh_MPa/km']) / 2 + \
                        (df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['cos_2beta']
    df_params['sigma_y_grad'] = (df_params['SH_MPa/km'] + df_params['Sh_MPa/km']) / 2 - \
                        (df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['cos_2beta']
    # tau_xy_grad should be positive after checking the directions of maximum stress in the CMG Results
    df_params['tau_xy_grad'] = -(df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['sin_2beta']

    # calculate the stress state for the reference block in kPa
    grid_ave = (ref_block_top_depth + ref_block_bottom_depth)/2
    df_params['sigma_x_ref'] = df_params['sigma_x_grad'] * grid_ave *(-1)
    df_params['sigma_y_ref'] = df_params['sigma_y_grad'] * grid_ave *(-1)
    df_params['sigma_z_ref'] = df_params['Sv_MPa/km'] * grid_ave *(-1)
    df_params['tau_xy_ref'] = df_params['tau_xy_grad'] * grid_ave *(-1)

    # Output
    df_params.to_csv(output_file_path/f"{name_prefix}_sampled_params_seed{random_seed}.csv", index=False,float_format='%.6g')
    

    if show_results:
        # Round the values for display
        def signif(x, sig=6):
            if not isinstance(x, (int, float, np.integer, np.floating)):
                return x
            if x == 0 or not np.isfinite(x):
                return x
            return round(x, sig - 1 - int(np.floor(np.log10(abs(x)))))

        df_display = df_params.copy()
        num_cols = df_display.select_dtypes(include=[np.number]).columns
        df_display[num_cols] = df_display[num_cols].map(lambda v: signif(v, 6))

        display(df_display)

    sampling_results = {
        'random_seed': random_seed,
        'name_prefix': name_prefix,
        # 'param_dataframe': df_params.round(2)
        'param_dataframe': df_params
    }

    return sampling_results

# perform Latin Hypercube Sampling and importance sampling for the Sula CCS and store the results in a CSV file
def latin_hypercube_and_importance_sampling(
    random_seed: int,
    name_prefix: str,
    property_file_names_path: str,
    output_file_path: str,
    n_samples: int,
    param_names: list,
    lower_bounds: list,
    upper_bounds: list,
    ref_block_top_depth: float,
    ref_block_bottom_depth: float,
    show_results: bool,
    # below is for importance sampling
    alpha: float,
    beta: float,
    proposal_SH_azi_low: float,
    proposal_SH_low: float,
    show_summary: bool = True
    ):

    # Load PORO and PERMX file names
    property_file_names = np.loadtxt(property_file_names_path,delimiter=",",dtype=str)

    # sort the file names by the number in the name
    def extract_number(filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else float('inf')

    poro_file_names = sorted(
        [name for name in property_file_names if "PORO" in name.upper()],
        key=extract_number
    )

    permx_file_names = sorted(
        [name for name in property_file_names if "PERMX" in name.upper()],
        key=extract_number
    )

    # check a few things
    if not poro_file_names or not permx_file_names:
        print("Error: PORO or PERMX file names not found.")
        sys.exit(1)

    if len(poro_file_names) != len(permx_file_names):
        raise ValueError(f"Number of PORO file names ({len(poro_file_names)}) does not match number of PERMX file names ({len(permx_file_names)})")

    num_pairs = len(poro_file_names)

    if n_samples > num_pairs:
        raise ValueError(f"Cannot sample {n_samples} unique poro/permx pairs: only {num_pairs} available.")

    # Latin Hypercube Sampling for parameters
    sampler = qmc.LatinHypercube(d=len(param_names), seed=random_seed)
    sample = sampler.random(n=n_samples)
    sample_params = qmc.scale(sample, lower_bounds, upper_bounds)
    df_params = pd.DataFrame(sample_params, columns=param_names)

    # Store poro/permx pairs
    df_params["PORO_file"] = [str(poro_file_names[i]) for i in range(n_samples)]
    df_params["PERMX_file"] = [str(permx_file_names[i]) for i in range(n_samples)]

    # add prefix to file names
    prefix = "data_properties/"
    df_params["PORO_file"] = df_params["PORO_file"].apply(lambda x: f"{prefix}{x}")
    df_params["PERMX_file"] = df_params["PERMX_file"].apply(lambda x: f"{prefix}{x}")

    ########################################## add importance sampling ###############
    # perform importanc sampling and save samples to a csv file
    IS_SH_azi_SH_uniform(
        random_seed = random_seed,
        name_prefix = name_prefix,
        output_file_path =  output_file_path,
        n_samples = n_samples,
        alpha = alpha,
        beta = beta,
        proposal_SH_azi_low = proposal_SH_azi_low,
        proposal_SH_low = proposal_SH_low,
        show_summary = show_summary
        )
    # load importance samples from the csv file
    importance_samples = pd.read_csv( output_file_path/f'{name_prefix}_importance_sampling.csv')
    df_params['SH_azi_deg'] = importance_samples['SH_azi_deg'].values
    df_params['SH_MPa/km'] = importance_samples['SH_MPa/km'].values
    ########################################## end of importance sampling ###############
    # Calculate stress state parameters
    df_params['beta'] = df_params['SH_azi_deg'] - 90  # Rotate from SH to x-axis
    df_params['cos_2beta'] = np.cos(np.radians(2 * df_params['beta']))
    df_params['sin_2beta'] = np.sin(np.radians(2 * df_params['beta']))

    # calculate the stress gradients in kPa/km
    df_params['sigma_x_grad'] = (df_params['SH_MPa/km'] + df_params['Sh_MPa/km']) / 2 + \
                        (df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['cos_2beta']
    df_params['sigma_y_grad'] = (df_params['SH_MPa/km'] + df_params['Sh_MPa/km']) / 2 - \
                        (df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['cos_2beta']
    # tau_xy_grad should be positive after checking the directions of maximum stress in the CMG Results
    df_params['tau_xy_grad'] = -(df_params['SH_MPa/km'] - df_params['Sh_MPa/km']) / 2 * df_params['sin_2beta']

    # calculate the stress state for the reference block in kPa
    grid_ave = (ref_block_top_depth + ref_block_bottom_depth)/2
    df_params['sigma_x_ref'] = df_params['sigma_x_grad'] * grid_ave *(-1)
    df_params['sigma_y_ref'] = df_params['sigma_y_grad'] * grid_ave *(-1)
    df_params['sigma_z_ref'] = df_params['Sv_MPa/km'] * grid_ave *(-1)
    df_params['tau_xy_ref'] = df_params['tau_xy_grad'] * grid_ave *(-1)

    # Output
    df_params.to_csv(output_file_path/f"{name_prefix}_sampled_params_seed{random_seed}.csv", index=False,float_format='%.2f')
    
    if show_results:
        display(df_params.round(2))

    sampling_results = {
        'random_seed': random_seed,
        'name_prefix': name_prefix,
        'param_dataframe': df_params.round(2)
    }

    return sampling_results

# Importance sampling for the Sula CCS
def IS_SH_azi_SH_uniform(
    random_seed: int,
    name_prefix: str,
    output_file_path: str,
    n_samples: int,
    alpha: float,
    beta: float,
    proposal_SH_azi_low: float,
    proposal_SH_low: float,
    show_summary: bool = True
    ):

    np.random.seed(random_seed)
    # name_prefix = '251023'
    # Parameters
    # n_samples = 90
    # alpha = 0.9
    # beta = 0.9

    # --- Target distributions ---
    # SH_azi ~ U(300, 320)
    target_SH_azi_low, target_SH_azi_high = 300, 320
    # proposal_SH_azi_low = 319
    p_SH_azi = 1/(target_SH_azi_high - target_SH_azi_low)

    # SH ~ U(16.2, 19.8)
    SH_base = 18
    target_SH_low, target_SH_high = SH_base * 0.9, SH_base * 1.1 # U(16.2,19.8)
    # proposal_SH_low = 18 * 1.05
    p_SH = 1/(target_SH_high - target_SH_low)

    # --- Proposal mixture components ---
    # SH_azi: 0.2 U(300,310) + 0.8 U(310,320)
    u1 = np.random.rand(n_samples)
    u1_quantile = np.quantile(u1,alpha)
    samples_SH_azi = np.empty(n_samples)
    samples_SH_azi[u1 > u1_quantile] = np.random.uniform(target_SH_azi_low, proposal_SH_azi_low, size=(u1 > u1_quantile).sum())
    samples_SH_azi[u1 <= u1_quantile] = np.random.uniform(proposal_SH_azi_low, target_SH_azi_high, size=(u1 <= u1_quantile).sum())
    q_SH_azi = np.where((samples_SH_azi >= proposal_SH_azi_low) & (samples_SH_azi < target_SH_azi_high), 
                alpha / (target_SH_azi_high-target_SH_azi_low), 
                (1 - alpha) / target_SH_azi_high-target_SH_azi_low)


    # SH: 0.2 U(16.2,17) + 0.8 U(17,19.8)
    u2 = np.random.rand(n_samples)
    u2_quantile = np.quantile(u2,beta)
    samples_SH = np.empty(n_samples)
    samples_SH[u2 > u2_quantile] = np.random.uniform(target_SH_low, proposal_SH_low, size=(u2 > u2_quantile).sum())
    samples_SH[u2 <= u2_quantile] = np.random.uniform(proposal_SH_low, target_SH_high, size=(u2 <= u2_quantile).sum())
    q_SH = np.where((samples_SH >= proposal_SH_low) & (samples_SH < target_SH_high), 
            beta / (target_SH_high-target_SH_low), 
            (1 - beta) / (target_SH_high-target_SH_low))

    # --- Importance weights ---
    # Since the two variables are independent, total weight = (p_SH_azi * p_SH) / (q_SH_azi*q_SH)
    weights = (p_SH_azi * p_SH) / (q_SH_azi * q_SH)
    weights /= np.sum(weights)

    # combine into a 2D array, note SH is turned into negative for CMG sign convention
    importance_sampling = np.column_stack((samples_SH_azi,q_SH_azi,-samples_SH,q_SH,weights))

    header_string = 'SH_azi_deg,q_SH_azi,SH_MPa/km,q_SH,weights'
    np.savetxt(output_file_path/f'{name_prefix}_importance_sampling.csv',importance_sampling,delimiter=',',fmt='%.4f',header=header_string,comments='')
    
    if show_summary:
        print('SH_azi')
        print(f'Target distribution: U[{target_SH_azi_low}, {target_SH_azi_high}], {n_samples} samples')
        print(f'Proposal distribution: {(1-alpha):.1f} * U[{target_SH_azi_low}, {proposal_SH_azi_low}] + {alpha} * U[{proposal_SH_azi_low}])')
        print(f'Importance samples min: {np.min(samples_SH_azi):.2f}, max: {np.max(samples_SH_azi):.2f}, number of alpha samples: {np.sum(importance_sampling[:,0] > proposal_SH_azi_low)}')
        print('SH')
        print(f'Target distribution: U[{target_SH_low}, {target_SH_high}], {n_samples} samples')
        print(f'Proposal distribution: {(1-beta):.1f} * U[{target_SH_low}, {proposal_SH_low:.2f}] + {beta} * U[{proposal_SH_low:.2f}, {target_SH_high}]')
        print(f'Importance samples min: {np.min(samples_SH):.2f}, max: {np.max(samples_SH):.2f}, number of beta samples: {np.sum(importance_sampling[:,2] < -proposal_SH_low)}')