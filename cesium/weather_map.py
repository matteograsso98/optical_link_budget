from herbie import Herbie, HerbieLatest
import matplotlib.pyplot as plt
import numpy as np

print("Fetching NOAA GFS data...")
# Initialize Herbie
H = HerbieLatest(model='gfs', product='pgrb2.0p25', fxx=0)

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


#—— WIND SPEED DATA —————————————————————————————————————————————————————————————--
try:
    ds_u = H.xarray("UGRD:10 m above ground")
    ds_v = H.xarray("VGRD:10 m above ground")
except FileNotFoundError:
    print("Error: Could not find wind variables. Double-check the search string.")
    exit()

lat = ds_u.latitude.values
lon = ds_u.longitude.values

u = ds_u[list(ds_u.data_vars)[0]].values
v = ds_v[list(ds_v.data_vars)[0]].values

# Wind speed magnitude [m/s]
wind_speed = np.sqrt(u**2 + v**2)

# GFS uses [0, 360] longitude — shift to [-180, 180]
lon_shifted = np.where(lon > 180, lon - 360, lon)
sort_idx = np.argsort(lon_shifted)
wind_speed = wind_speed[:, sort_idx]


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


#—— WIND SPEED MAP —————————————————————————————————————————————————————————————--
print("Generating wind speed overlay...")
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# bleu = calme, rouge = vent fort (vmax=30 m/s)
ax.imshow(wind_speed, cmap='RdYlBu_r', extent=[-180, 180, -90, 90],
          origin='upper', vmin=0, vmax=30)

plt.savefig('wind_speed_map.png', transparent=True)
print("✅ Saved wind_speed_map.png!")