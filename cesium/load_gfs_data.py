"""GFSData — stateful GFS field loader via Herbie."""

import numpy as np
from herbie import HerbieLatest


class GFSData:
    """Loads GFS fields, returning 2-D arrays already shifted to [-180, 180]."""

    def __init__(self, model="gfs", product="pgrb2.0p25", fxx=0):
        self.H = HerbieLatest(model=model, product=product, fxx=fxx)
        # Grid geometry is identical across fields; resolved lazily on first fetch.
        self.lat = None
        self.lon = None
        self._sort_idx = None

    def field(self, search):
        """Return the first variable matching `search` as a lon-sorted 2-D array."""
        try:
            ds = self.H.xarray(search)
        except FileNotFoundError:
            raise FileNotFoundError(f"No GFS message matched '{search}'.")
        var = ds[list(ds.data_vars)[0]].values

        if self._sort_idx is None:
            lon = np.where(ds.longitude.values > 180,
                           ds.longitude.values - 360, ds.longitude.values)
            self._sort_idx = np.argsort(lon)
            self.lon = lon[self._sort_idx]
            self.lat = ds.latitude.values

        return var[:, self._sort_idx]

    def wind_speed(self, level):
        """Wind-speed magnitude [m/s] at e.g. '10 m', '50 m', '100 m'."""
        u = self.field(f"UGRD:{level} above ground")
        v = self.field(f"VGRD:{level} above ground")
        return np.sqrt(u**2 + v**2)
