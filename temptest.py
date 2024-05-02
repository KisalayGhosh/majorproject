import netCDF4 as nc

# Path to the GRD file
grd_file_path = 'temparature.GRD'

# Open the GRD file
try:
    grd_dataset = nc.Dataset(grd_file_path, 'r')
except Exception as e:
    print(f"Error opening GRD file: {e}")
    exit(1)

# Print schema
print("Schema for GRD file:")
for var_name in grd_dataset.variables:
    var = grd_dataset.variables[var_name]
    print(f"Variable: {var_name}")
    print(f"Dimensions: {var.dimensions}")
    print("Attributes:")
    for attr_name in var.ncattrs():
        print(f" - {attr_name}: {getattr(var, attr_name)}")
    print()

# Close the GRD file
grd_dataset.close()
