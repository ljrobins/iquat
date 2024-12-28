import mirage as mr
import numpy as np

lat_geod_deg = 40
lon_deg = 40


mr.tic()
date = mr.now()
itrf_pos = mr.lla_to_itrf(
    lat_geod=np.deg2rad(lat_geod_deg), lon=np.deg2rad(lon_deg), alt_km=0.0
)
j2000_pos = mr.itrf_to_j2000(date) @ itrf_pos
mr.toc()

mr.tic()
date = mr.now()
itrf_pos = mr.lla_to_itrf(
    lat_geod=np.deg2rad(lat_geod_deg), lon=np.deg2rad(lon_deg), alt_km=0.0
)
j2000_pos = mr.itrf_to_j2000(date) @ itrf_pos
mr.toc()
print(j2000_pos)

mr.tic()
station = mr.Station(
    lat_deg=lat_geod_deg, lon_deg=lon_deg, altitude_reference="geoid", alt_km=0.0
)
j2000_pos = station.j2000_at_dates(mr.now())
mr.toc()

mr.tic()
station = mr.Station(
    lat_deg=lat_geod_deg, lon_deg=lon_deg, altitude_reference="geoid", alt_km=0.0
)
j2000_pos = station.j2000_at_dates(mr.now())
mr.toc()

print(station.geoid_height_km)
print(station.itrf)
print(itrf_pos)
print(j2000_pos)
