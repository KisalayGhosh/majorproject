import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to the NetCDF file
nc_file_path = 'rainfall.nc'
if not os.path.isfile(nc_file_path):
    print(f"File '{nc_file_path}' not found.")

# Create a directory to save the images
output_dir = 'rainfall_images'
os.makedirs(output_dir, exist_ok=True)

# Open the NetCDF file
try:
    nc_file = nc.Dataset(nc_file_path, 'r')
except Exception as e:
    print(f"Error opening NetCDF file: {e}")
    exit(1)

# Access image data
image_data = nc_file.variables['RAINFALL']  # Assuming 'RAINFALL' is the variable containing image data

# Get the number of time steps
num_time_steps = image_data.shape[0]

# Define the sequence length (you can adjust this based on your requirements)
sequence_length = 5

# Iterate over each time step to create sequences of images
for i in range(num_time_steps - sequence_length + 1):
    # Initialize an empty list to store images for the sequence
    sequence_images = []
    
    # Extract images for the current sequence
    for j in range(sequence_length):
        # Get the 2D array (image) for the current time step
        image_array = image_data[i + j, :, :]  # Assuming the dimensions are (time, latitude, longitude)
        sequence_images.append(image_array)
    
    # Convert the list of images to a numpy array
    sequence_images = np.array(sequence_images)
    
    # Plot the sequence of images
    fig, axes = plt.subplots(1, sequence_length, figsize=(4 * sequence_length, 4))
    for j, image_array in enumerate(sequence_images):
        axes[j].imshow(image_array, cmap='viridis')  # Use any colormap you prefer
        axes[j].set_title(f'Day {i + j + 1}')
        axes[j].axis('off')
    plt.tight_layout()
    
    # Save the plot as an image file (PNG format)
    output_file = os.path.join(output_dir, f'rainfall_sequence_{i + 1}.png')
    plt.savefig(output_file)
    plt.close()  # Close the plot to release memory

# Close the NetCDF file
nc_file.close()

print("Images saved successfully.")
