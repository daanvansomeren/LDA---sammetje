import csv
import glob
import os
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from lmfit import models


fit_parameters_dataframe = pd.DataFrame({"Amplitude": [], "mu": [], "sigma": []})
def replace_chars_in_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Read the file's content as text
        content = infile.read()
        
        # Replace commas with dots and semicolons with commas
        modified_content = content.replace(',', '.').replace(';', ',').replace('\t',',')
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write the modified content to the new file
        outfile.write(modified_content)

def make_csv(dagdeel, promille):
    directory = r'C:\documenten computer\huiswerkcomputer\Natuur en sterrenkunde jaar 2\project natuur en sterrenkunde 2\ECPC\LDA---sammetje\csv'
    output_directory = r'C:\documenten computer\huiswerkcomputer\Natuur en sterrenkunde jaar 2\project natuur en sterrenkunde 2\ECPC\LDA---sammetje\output'

    file_pattern = os.path.join(directory, '*.csv')
    
    # Get the list of matching files
    csv_files = glob.glob(file_pattern)
    
    # Count the files
    dir_len = len(csv_files)
    print(f"Found {dir_len} files in the directory.")

    for meting in range(1, dir_len + 1):
        # Construct input and output file paths
        input_file = os.path.join(directory, f"{dagdeel}meting{meting}_{promille}.csv")
        output_file = os.path.join(output_directory, f"{dagdeel}meting{meting}_{promille}_nieuw.csv")
        
        # Check if the input file exists
        if os.path.exists(input_file):
            replace_chars_in_csv(input_file, output_file)
            print(f"Processed: {input_file} -> {output_file}")
            fit(output_file, f"{dagdeel}meting{meting}_{promille}.csv")
        else:
            print(f"File not found: {input_file}")
    os.makedirs(r'C:\documenten computer\huiswerkcomputer\Natuur en sterrenkunde jaar 2\project natuur en sterrenkunde 2\ECPC\LDA---sammetje\csv_parameters', exist_ok=True)  
    fit_parameters_dataframe.to_csv(r'C:\documenten computer\huiswerkcomputer\Natuur en sterrenkunde jaar 2\project natuur en sterrenkunde 2\ECPC\LDA---sammetje\csv_parameters/1para_15.csv')
def gaussische_functie(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))


def bins(input_file):
    df = pd.read_csv(input_file)
    df_clean = df.dropna(subset=[df.columns[-2], df.columns[-1]])
    frequentie = df_clean.iloc[:, -2].tolist()  # First list (last but one column)
    amplitude = df_clean.iloc[:, -1].tolist()  # Second list (last column)
    return frequentie, amplitude

def fit(input_file, naam):
    global fit_parameters_dataframe  # Declare that we are modifying the global variable
    frequentie, amplitude = bins(input_file)
    
    # Custom peak amplitude for specific file
    
    if naam == "4meting11_15.csv" or naam == "4meting15_15.csv" or naam == "4meting16_15.csv":
        x = 5000
        y = 100
    elif naam == "4meting17_15.csv" or naam == "4meting18_15.csv":
        x = 2300
        y = 100
    else:
        x = frequentie[np.argmax(amplitude)]
        y = max(amplitude)

    '''
    if naam == "4meting15_2_25.csv":
        x = 600
        y = 200
    else:
        x = frequentie[np.argmax(amplitude)]
        y = max(amplitude)
        '''
    print ("x =", x)
    print ("y =", y)

    # Gaussian fitting
    popt, pcov = curve_fit(
        gaussische_functie, 
        frequentie, 
        amplitude, 
        p0=[y, x, 60]
    )
    A_fit, mu_fit, sigma_fit = popt
    
    # Generate fit curve
    x_fit = np.linspace(frequentie[0], frequentie[-1], 500)
    y_fit = gaussische_functie(x_fit, A_fit, mu_fit, sigma_fit)

    # Update the global DataFrame with new fit parameters
    nieuwe_rij = pd.DataFrame({"Amplitude": [A_fit], "mu": [mu_fit], "sigma": [sigma_fit]})
    fit_parameters_dataframe = pd.concat([fit_parameters_dataframe, nieuwe_rij], ignore_index=True)

    # Plotting
    plt.bar(frequentie, amplitude, width=10, alpha=0.6, label='Histogram Data')
    plt.plot(x_fit, y_fit, 'r-', label=f'Gaussian Fit: $A={A_fit:.2f}, \mu={mu_fit:.2f}, \sigma={sigma_fit:.2f}$')
    plt.legend()
    plt.xlabel('Range')
    plt.ylabel('Amplitude')
    plt.title(f"Histogram and Gaussian Fit - meting {naam}")
    plt.show()


make_csv(4, "2_25")
