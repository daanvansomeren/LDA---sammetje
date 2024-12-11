import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
import math

# allereerste testmeting 
testmeting = [
    [],
    [2526.64, 3996.63, 5298.98, 6006.13, 5827.50, 4067.95, 3444.02],
    []
]
# nulmeting 0 % PEO
nulmeting = [
    [],
    [1348.54, 2128.13, 2866.26, 3468.28, 3982.77, 4358.45, 4742.03, 4910.19, 5104.66, 5119.41, 4967.07, 4780.89, 4627.36, 4275.79, 3906.08, 3180.87, 2410.09, 1919.25],
    []
]

# eerste PEO meting; 0.1 % PEO
prom_1 = [
    [],
    [1033.41, 1393.76, 2205.81, 2692.46, 3420.27, 4105.12, 4305.08, 4693.49, 4991.62, 5187.31, 5223.11, 5272.89,  5265.73, 5262.89, 5229.82, 5007.03, 3124.70, 2845.03, 2749.52, 2731.38, 2506.40],
    []
]

# tweede PEO meting; 0.5 % PEO 
prom_5_1 = [
    [1493.48, 1584.23, 1757.23],
    [2131.49, 2715.75, 3298.19, 3799.56, 4267.68, 4662.67, 5024.63, 5336.62, 5632.79, 5763.05, 5869.16, 5617.22, 5102.27, 4574.41, 4189.78, 2675.74],
    [2451.65, 1845.22, 1703.20, 1435.68]
]
# derde PEO meting; 0.5 % PEO
prom_5_2 = [
    [1123.58, 1564.17, 1829.12],
    [2312.93, 3000.25, 3598.51, 4088.18, 4555.19, 4919.56, 5210.15, 5480.11, 5673.19, 5804.4, 5690.16, 5413.31, 5135.43],
    []
]

# vierde PEO meting; 1 % PEO
prom_10 = [
    [1480.96, 1948.64, 2324.11],
    [2800.39, 3572.22, 4195.38, 4714.98, 5062.35, 5343.79, 5517.24, 5598.99, 5704, 5607.5, 5463.54, 5114.49, 4635.49, 3939.6, 2760.36],
    []
]




# Define the datasets
datasets = {
    "Nulmeting": nulmeting,
    "Prom 0.1% PEO": prom_1,
    "Prom 0.5% PEO (1)": prom_5_1,
    "Prom 0.5% PEO (2)": prom_5_2,
    "Prom 1% PEO": prom_10
}

# Define colors for each dataset
colors = {
    "Nulmeting": "orange",
    "Prom 0.1% PEO": "green",
    "Prom 0.5% PEO (1)": "red",
    "Prom 0.5% PEO (2)": "purple",
    "Prom 1% PEO": "black"
    
}



testkeuze = prom_5_2

def r_lijst(testkeuze):
    half_o, heel, half_b = testkeuze
    x = len(heel) / 2
    r = np.arange(-x, x, 1)
    r_half_o = [-x - 0.5 - 0.5 * i for i in reversed(range(len(half_o)))]
    r_half_b = [x + 0.5 * i for i in range(len(half_b))]
    r = r_half_o + list(r) + r_half_b
    metingen = half_o + heel + half_b
    return r, metingen

def r_lijst_gespiegeld(testkeuze):
    half_o, heel, half_b = testkeuze
    print(heel)
    Hz_old = 0
    metingen = half_o + heel
    for position in range (0, len(metingen) + 1):
        if Hz_old > heel[position]:
            heel_nieuw = heel[:position - 1]
            r = np.arange(-position + 2, 1, 1)
            r_half_o = [-position + 1.5 - 0.5 * i for i in reversed(range(len(half_o)))]
            r = r_half_o + list(r)
            metingen = half_o + heel_nieuw
            return spiegel_grafiek(r, metingen)
        else:
            Hz_old = metingen[position]
    

r, metingen = r_lijst(testkeuze)

def spiegel_grafiek(r, metingen):
    r_mirrored = [-value for value in r]
    r_combined = np.concatenate((r, r_mirrored))
    metingen_combined = np.concatenate((metingen, metingen))
    sorted_indices = np.argsort(r_combined)
    r_sorted = np.array(r_combined)[sorted_indices]
    metingen_sorted = np.array(metingen_combined)[sorted_indices]
    
    return r_sorted, metingen_sorted

r, metingen = r_lijst_gespiegeld(testkeuze)



# Define the quadratic function
def snelheid_functie(x, a, b):
    return a * (x ** 2) + b

# Fit the model
ons_model = models.Model(snelheid_functie)
result = ons_model.fit(metingen, x=r, weights=None, a=1, b=2000)

# Print fit report
print(result.fit_report())

# Generate 200 evenly spaced r values
r_200 = np.linspace(min(r), max(r), 200)

# Generate fitted y values (y_fit) for 200 points
y_fit_200 = [snelheid_functie(val, result.params['a'].value, result.params['b'].value) for val in r_200]


#calculates length between mirrors to angle
diff = 22
angle = np.arctan((diff/2)/130) * 1.33
print(f"angle is {angle}" )
# Calculate speeds for 200 points using sin in radians
sin_value = np.sin(angle)  # 0.0805 is already in radians
speeds_200 = [632.8 * y / (2 * sin_value) / 1000000 for y in y_fit_200]

# Verify the highest speed
max_speed = max(speeds_200)
max_y_fit = max(y_fit_200)
print(f"Maximum y_fit: {max_y_fit}, Maximum Speed: {max_speed}")

# Plot the speeds for 200 points
plt.figure()  # Create a new figure for speeds
plt.plot(r_200, speeds_200, '-', label='Calculated Speeds (200 points)', color='green')
plt.xlabel('r (mm)')
plt.ylabel('Speed (mm/s)')
plt.title('Speed Calculated from Fitted Values')
plt.legend()
plt.show()

# Plot the original data points
plt.figure(figsize=(8, 6))
plt.plot(r, metingen, 'o', label='Original Data Points', color='blue')

# Plot the quadratic fit
plt.plot(r_200, y_fit_200, '-', label='Quadratic Fit', color='red')

# Add labels, title, and legend
plt.xlabel('r')
plt.ylabel('positie_fit')
plt.title('Quadratic Fit of Data')
plt.legend()
plt.grid()

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))

# Loop through each dataset and perform fitting
for label, dataset in datasets.items():
    r_mirrored, metingen_mirrored = r_lijst(dataset)  # Get r and metingen for the dataset

    # Fit the model
    ons_model = models.Model(snelheid_functie)
    result = ons_model.fit(metingen_mirrored, x=r_mirrored, weights=None, a=1, b=2000)

    # Generate fitted y values for plotting
    r_fine = np.linspace(min(r_mirrored), max(r_mirrored), 200)
    y_fit = [snelheid_functie(val, result.params['a'].value, result.params['b'].value) for val in r_fine]

    # Plot the data points and the fit
    plt.plot(r_mirrored, metingen_mirrored, 'o', label=f"{label} Data", color=colors[label], alpha=0.6)
    plt.plot(r_fine, y_fit, '-', label=f"{label} Fit", color=colors[label])

# Add labels, title, legend, and grid
plt.xlabel('r')
plt.ylabel('metingen')
plt.title('Quadratic Fits for All Datasets')
plt.legend()
plt.grid()

# Show the plot
plt.show()