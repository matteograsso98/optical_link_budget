from herbie import Herbie, HerbieLatest
import matplotlib.pyplot as plt
import numpy as np

print("Fetching NOAA GFS data...")
# Initialize Herbie
H = HerbieLatest(model='gfs', product='pgrb2.0p25', fxx=0)


# Retrieve the complete inventory as a Pandas DataFrame
df_inventory = H.inventory()

# Extract and display only the single list of available variables
# The 'search_this' or 'param' column contains the variable identifier
variables = df_inventory["search_this"].unique()

print(f"Total number of messages/variables : {len(df_inventory)}")
print("\n--- Abbreviated list of available variables ---")
for var in variables:  # Displays the first 20 for this example
    print(var)

exit()  # Stop the script after displaying the list of variables


#—— TEMPERATURE DATA —————————————————————————————————————————————————————————————--
try:
    ds = H.xarray("TMP:2 m above ground")
except FileNotFoundError:
    print("Error: Could not find the subset. Double-check your search string.")
    exit()

# Extract Xarray data
temp_k = ds.t2m.values
lat = ds.latitude.values
lon = ds.longitude.values

# Convert Kelvin to Celsius
temp_c = temp_k - 273.15

# NOAA GFS uses [0, 360] longitude. 
# Shift to standard [-180, 180] mapping
lon_shifted = np.where(lon > 180, lon - 360, lon)

# THE FIX: Sort the 1D longitude array
sort_idx = np.argsort(lon_shifted)

# Apply the sorting index to the longitude axis (columns) of the temperature array
temp_c = temp_c[:, sort_idx]


#—— WIND SPEED DATA (10 m above ground) —————————————————————————————————————————————————————————————--
try:
    ds_u_10m = H.xarray("UGRD:10 m above ground")
    ds_v_10m = H.xarray("VGRD:10 m above ground")
except FileNotFoundError:
    print("Error: Could not find wind variables. Double-check the search string.")
    exit()

lat = ds_u_10m.latitude.values
lon = ds_u_10m.longitude.values

u_10m = ds_u_10m[list(ds_u_10m.data_vars)[0]].values
v_10m = ds_v_10m[list(ds_v_10m.data_vars)[0]].values

# Wind speed magnitude [m/s]
wind_speed_10m = np.sqrt(u_10m**2 + v_10m   **2)

# GFS uses [0, 360] longitude — shift to [-180, 180]
lon_shifted = np.where(lon > 180, lon - 360, lon)
sort_idx = np.argsort(lon_shifted)
wind_speed_10m = wind_speed_10m[:, sort_idx]


#—— WIND SPEED DATA (50 m above ground) —————————————————————————————————————————————————————————————--
try:
    ds_u_50m = H.xarray("UGRD:50 m above ground")
    ds_v_50m = H.xarray("VGRD:50 m above ground")
except FileNotFoundError:
    print("Error: Could not find wind variables. Double-check the search string.")
    exit()

lat = ds_u_50m.latitude.values
lon = ds_u_50m.longitude.values

u_50m = ds_u_50m[list(ds_u_50m.data_vars)[0]].values
v_50m = ds_v_50m[list(ds_v_50m.data_vars)[0]].values

# Wind speed magnitude [m/s]
wind_speed_50m = np.sqrt(u_50m**2 + v_50m**2)

# GFS uses [0, 360] longitude — shift to [-180, 180]
lon_shifted = np.where(lon > 180, lon - 360, lon)
sort_idx = np.argsort(lon_shifted)
wind_speed_50m = wind_speed_50m[:, sort_idx]


#—— TEMPERATURE MAP —————————————————————————————————————————————————————————————--
print("Generating map overlay...")
# Setup a Matplotlib figure with NO margins, borders, or axes.
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Map the data to a global bounding box
ax.imshow(temp_c, cmap='turbo', extent=[-180, 180, -90, 90], origin='upper')

# Save as a transparent PNG
plt.savefig('temperature_map.png', transparent=True)
print("✅ Saved temperature_map.png!")


#—— WIND SPEED MAP (10 m above ground) ——————————————————————————————————————————————————————————————--
print("Generating wind speed overlay...")
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# bleu = calme, rouge = vent fort (vmax=30 m/s)
ax.imshow(wind_speed_10m, cmap='RdYlBu_r', extent=[-180, 180, -90, 90],
          origin='upper', vmin=0, vmax=30)

plt.savefig('wind_speed_10m_map.png', transparent=True)
print("✅ Saved wind_speed_10m_map.png!")


#—— WIND SPEED DATA (100 m above ground) —————————————————————————————————————————————————————————————--
try:
    ds_u_100m = H.xarray("UGRD:100 m above ground")
    ds_v_100m = H.xarray("VGRD:100 m above ground")
except FileNotFoundError:
    print("Error: Could not find 100m wind variables. Double-check the search string.")
    exit()

u_100m = ds_u_100m[list(ds_u_100m.data_vars)[0]].values
v_100m = ds_v_100m[list(ds_v_100m.data_vars)[0]].values

wind_speed_100m = np.sqrt(u_100m**2 + v_100m**2)
lon_shifted = np.where(ds_u_100m.longitude.values > 180, ds_u_100m.longitude.values - 360, ds_u_100m.longitude.values)
wind_speed_100m = wind_speed_100m[:, np.argsort(lon_shifted)]


#—— WIND SPEED MAP (50 m above ground) ——————————————————————————————————————————————————————————————--
print("Generating wind speed overlay...")
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(wind_speed_50m, cmap='RdYlBu_r', extent=[-180, 180, -90, 90],
          origin='upper', vmin=0, vmax=30)

plt.savefig('wind_speed_50m_map.png', transparent=True)
print("✅ Saved wind_speed_50m_map.png!")


#—— WIND SPEED MAP (100 m above ground) ——————————————————————————————————————————————————————————————--
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(wind_speed_100m, cmap='RdYlBu_r', extent=[-180, 180, -90, 90],
          origin='upper', vmin=0, vmax=30)

plt.savefig('wind_speed_100m_map.png', transparent=True)
print("✅ Saved wind_speed_100m_map.png!")