import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import os
import pandas as pd
from matplotlib.gridspec import GridSpec

# HANDMATIGE DATASETS

#Datasets
nulmeting = [
    [1348.54, 2128.13, 2866.26, 3468.28, 3982.77, 4358.45, 4742.03, 4910.19, 5104.66, 
     5119.41, 4967.07, 4780.89, 4627.36, 4275.79, 3906.08, 3180.87, 2410.09, 1919.25], 
    [-8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
]

prom_1 = [
    [2205.81, 2692.46, 3420.27, 4105.12, 4305.08, 4693.49, 4991.62, 
     5187.31, 5223.11, 5272.89, 5265.73, 5262.89, 5229.82, 5007.03, 3124.70, 2845.03, 
     2749.52, 2731.38, 2506.40], 
    [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
]

prom_5_1 = [
    [1584.23, 1757.23, 2131.49, 2715.75, 3298.19, 3799.56, 4267.68, 4662.67, 5024.63, 
     5336.62, 5632.79, 5763.05, 5869.16, 5617.22, 5102.27, 4574.41, 
     4189.78, 2675.74, 2451.65, 1845.22, 1703.20, 1435.68], 
    [-9, -8.5, -8, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9, 9.5]
]

prom_5_2 = [
    [1564.17, 1829.12, 2312.93, 3000.25, 3598.51, 4088.18, 4555.19, 4919.56, 5210.15, 
     5480.11, 5673.19, 5804.4, 5690.16, 5413.31, 5135.43], 
    [-9.5, -9, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5 , -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
]

prom_10_1 = [
    [1948.64, 2324.11, 2800.39, 3572.22, 4195.38, 4714.98, 5062.35, 5343.79, 5517.24, 
     5598.99, 5704, 5607.5, 5463.54, 5114.49, 4635.49, 3939.6, 2760.36], 
    [-9.5, -9, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
]

prom_10_2 = [
    [1823.06, 2350.61, 3171.78, 3914.95, 4492.56, 4933.78, 5309.12, 5569.46, 5701.36, 
     5702.45, 5610.35, 5558.56, 5278.59, 4885.61, 4776.75, 4458.98, 3330.29], 
    [-8.5, -8, -7.5, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
]

prom_10_3 = [
    [1765.72, 2183.47, 2563.43, 3418.32, 4100.53, 4627.59, 4831.69, 5366.79, 5385.11, 5659.99, 5745.05, 5689.92, 5642.19, 5277.96, 4926.35, 4505.28, 4002.05, 3189.2, 2408.98, 2060.97, 1846.17], 
    [-9.5, -9, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5]
]

prom_01 = [
    [1173.05, 1847.92, 2807.57, 3522.97, 4108.93, 4582.90, 4837.12, 5043.80, 5085.97, 5087.94, 4880.10, 4691.22, 4433.57, 4057.20, 3617.93, 2325.12, 1761.70, 1316.02],
    [-8.25, -7.75, -7.25, -6.75, -5.75, -4.75, -3.75, -2.75, -1.75, -0.75, 0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25]
]


list1 = [-9 + i * 0.5 for i in range(5)]  # 5 steps of 0.5 starting from -9
list2 = [list1[-1] + (i + 1) * 1 for i in range(11)]  # 11 steps of 1
list3 = [list2[-1] + (i + 1) * 0.5 for i in range(6)]  # 6 steps of 0.5

# Combining all parts of the list
final_list = list1 + list2 + list3
prom_15 = [
    [1301.41, 1712.72, 2145.49, 2554.71, 3294.96, 3920.26, 4220.05, 4505.18, 4806.74, 5094.10, 5251.48, 5304.78, 5335.8, 5228.71, 4844.62, 4332.33, 2501.41, 2441.17, 2258.31, 2035.56, 1815.98, 1606.19],
    final_list
]

hand_datasets = {
    "Nulmeting - 0‰ PEO": nulmeting,
    "0.1‰ PEO": prom_01,
    "1‰ PEO": prom_1,
    "5‰ PEO (1)": prom_5_1,
    "5‰ PEO (2)": prom_5_2,
    "10‰ PEO (1)": prom_10_1,
    "10‰ PEO (2)": prom_10_2,
    "10‰ PEO (3)": prom_10_3,
    "15‰ PEO": prom_15,
}

hand_colors = {
    "Nulmeting - 0‰ PEO": "orange",
    "0.1‰ PEO": "grey",
    "1‰ PEO": "green",
    "5‰ PEO (1)": "red",
    "5‰ PEO (2)": "purple",
    "10‰ PEO (1)": "magenta",
    "10‰ PEO (2)": "yellow",
    "10‰ PEO (3)": "cyan",
    "15‰ PEO": "blue",
}

# HANDMATIGE DATASETS

def df_read(meting_nummer, promille):
    directory = r'C:\Users\daanv\Documents\UVA\LDA - sammetje\csv_parameters'
    input_file = os.path.join(directory, f"{meting_nummer}para_{promille}.csv")
    df = pd.read_csv(input_file)
    amplitude = df.iloc[:, -3].tolist()  
    mu = df.iloc[:, -2].tolist() 
    sigma = df.iloc[:, -1].tolist()
    return amplitude, mu, sigma

prom_15 = [
    df_read(1, 15)[1], # freq
    [-9 + i for i in range(len(df_read(1, 15)[1]))], # r
    df_read(1, "15")[2], # sigma
]

prom_25_2 = [
    df_read(2, "25")[1], # freq
    [-9 + i for i in range(len(df_read(2, "25")[1]))], # r
    df_read(2, "25")[2] # sigma
]

datasets = {
    "15‰ PEO": prom_15,
    "25‰ PEO": prom_25_2
}

colors = {
    "15‰ PEO": "black",
    "25‰ PEO": "pink"
}

def fouten_prop(frequentie, sigma):
    golflengte = 632.8e-6  # Wavelength of the laser (in mm)
    hoek = 4.618 * np.pi / 180  # Convert angle to radians
    fout_hoek = 0.219 * np.pi / 180  # Convert angle uncertainty to radians
    form = (
        ((-golflengte * frequentie * fout_hoek) / (2 * (np.cos(hoek) ** 2))) ** 2
        + ((golflengte * sigma) / (2 * np.sin(hoek))) ** 2
    )
    return np.sqrt(form)

def quadratisch_functie(x, a, b, x0):
    return a * (x - x0) ** 2 + b

def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def v_functie(frequentie):
    form = (632.8e-6 * frequentie) / (2 * np.sin(4.618 * np.pi / 180))
    return form

def process_and_fit_with_residuals(dataset, label, color, fit_functie, ax_fit, ax_residual):
    metingen = dataset[0]
    r = [val * 1.33 for val in dataset[1]]
    fout_lijst = []
    snelheid = []
    
    if len(dataset) > 2:
        sigma_csv = dataset[2]  
        for pos in range(len(metingen)):
            fout_lijst.append(fouten_prop(metingen[pos], sigma_csv[pos]))
            snelheid.append(v_functie(metingen[pos]))
    else:
        sigma_csv = None
        for pos in range(len(metingen)):
            snelheid.append(v_functie(metingen[pos]))
    # Fit the model
    model = models.Model(fit_functie)
    if fit_functie == quadratisch_functie:
        result = model.fit(snelheid, x=r, a=0, b=np.mean(snelheid), x0=0)
        a = result.params['a'].value
        b = result.params['b'].value
        x0 = result.params['x0'].value

    if fit_functie == gauss:
        result = model.fit(snelheid, x=r, H=np.mean(snelheid), A=1, x0=0, sigma=100)
        H = result.params['H'].value
        A = result.params['A'].value
        x0 = result.params['x0'].value
        sigma = result.params['sigma'].value
    
    # Calculate the shift to align the peak (x0) to zero
    shift = -x0
    r_shifted = [val + shift for val in r]
    
    # Generate the shifted fit
    r_fine_shifted = np.linspace(min(r_shifted), max(r_shifted), 200)
    if fit_functie == quadratisch_functie:
        y_fit_shifted = fit_functie(r_fine_shifted, a, b, 0)
    if fit_functie == gauss:
        y_fit_shifted = fit_functie(r_fine_shifted, H, A, 0, sigma)
    
    if sigma_csv is not None:
        ax_fit.errorbar(r_shifted, snelheid, yerr=fout_lijst, fmt='o', label=f"{label} Meting", color=color, alpha=0.6)
    else:
        ax_fit.plot(r_shifted, snelheid, 'o', label=f"{label} Meting", color=color, alpha=0.6)
    ax_fit.plot(r_fine_shifted, y_fit_shifted, '-', label=f"{label} Fit", color=color)
    
    # Calculate residuals
    residuals = snelheid - result.eval(x=r)
    ax_residual.plot(r_shifted, residuals, 'o', label=f"{label} Residuals", color=color)
    ax_residual.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax_residual.set_xlabel("Afstand (mm)")
    ax_residual.set_ylabel("Snelheid (mm/s)")
def plot_combined(datasets_, colors, functie):
    num_datasets = len(datasets_)
    num_cols = 2  # Divide residuals across two columns
    num_rows = (num_datasets + 1) // 2  # Calculate required rows for residuals
    # Create a figure and a GridSpec layout
    fig = plt.figure(figsize=(19, 2 * num_rows))  # Adjusted figure size for better spacing
    gs = GridSpec(num_rows, num_cols + 3, figure=fig)  # Grid with extra columns for space adjustments
    # Create the shared axis for fits
    ax_fit = fig.add_subplot(gs[:, :2])  # First subplot spans two columns
    ax_fit.set_xlabel('Afstand (mm)')
    ax_fit.set_ylabel('Snelheid (mm/s)')
    ax_fit.grid()
    # Create residual axes
    residual_axes = []
    for i in range(num_datasets):
        col = i % num_cols + 3  # Start residual plots from the third column
        row = i // num_cols
        ax_residual = fig.add_subplot(gs[row, col])  # Adjust for residuals in individual subplots
        residual_axes.append(ax_residual)
    # Plot data and fits
    for i, (label, dataset) in enumerate(datasets_.items()):
        process_and_fit_with_residuals(dataset, label, colors[label], functie, ax_fit, residual_axes[i])
    # Add legends to the fits subplot
    ax_fit.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Tighten layout for better spacing
    plt.tight_layout()
    plt.show()

# Plot both datasets with residuals
plot_combined(datasets, colors, quadratisch_functie)