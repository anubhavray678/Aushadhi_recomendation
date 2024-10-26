import numpy as np
import matplotlib.pyplot as plt

# Define triangular membership function
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

# Define ranges
x_fever = np.linspace(35, 42, 500)
x_cough = np.linspace(0, 10, 500)
x_fatigue = np.linspace(0, 10, 500)

# Membership functions for Fever
fever_mild = triangular(x_fever, 36, 37, 38)
fever_moderate = triangular(x_fever, 37, 38, 39)
fever_severe = triangular(x_fever, 38, 40, 41)

# Membership functions for Cough
cough_mild = triangular(x_cough, 0, 2, 4)
cough_moderate = triangular(x_cough, 3, 5, 7)
cough_severe = triangular(x_cough, 6, 8, 10)

# Membership functions for Fatigue
fatigue_mild = triangular(x_fatigue, 0, 2, 4)
fatigue_moderate = triangular(x_fatigue, 3, 5, 7)
fatigue_severe = triangular(x_fatigue, 6, 8, 10)

# Plotting membership functions
plt.figure(figsize=(18, 10))

# Fever Membership
plt.subplot(3, 1, 1)
plt.plot(x_fever, fever_mild, label='Mild', color='b')
plt.plot(x_fever, fever_moderate, label='Moderate', color='g')
plt.plot(x_fever, fever_severe, label='Severe', color='r')
plt.title('Fever Membership Function')
plt.xlabel('Temperature (°C)')
plt.ylabel('Membership Degree')
plt.legend()

# Cough Membership
plt.subplot(3, 1, 2)
plt.plot(x_cough, cough_mild, label='Mild', color='b')
plt.plot(x_cough, cough_moderate, label='Moderate', color='g')
plt.plot(x_cough, cough_severe, label='Severe', color='r')
plt.title('Cough Membership Function')
plt.xlabel('Cough Level (0-10)')
plt.ylabel('Membership Degree')
plt.legend()

# Fatigue Membership
plt.subplot(3, 1, 3)
plt.plot(x_fatigue, fatigue_mild, label='Mild', color='b')
plt.plot(x_fatigue, fatigue_moderate, label='Moderate', color='g')
plt.plot(x_fatigue, fatigue_severe, label='Severe', color='r')
plt.title('Fatigue Membership Function')
plt.xlabel('Fatigue Level (0-10)')
plt.ylabel('Membership Degree')
plt.legend()

plt.tight_layout()
plt.show()
