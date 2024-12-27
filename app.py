from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import mirage as mr
import numpy as np
from pygeomag import GeoMag

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/", methods=["GET"])
def home():
    return render_template("quat.html")

station = None;

@socketio.on('time_update')
def handle_time_update(data):
    unix_time = int(data.get('unix_time'))
    date = mr.utc(1970, 1, 1) + mr.seconds(unix_time / 1000)
    jd = mr.date_to_jd(date)
    mjd = mr.jd_to_mjd(jd)
    gmst = mr.date_to_gmst(date)
    # print(date.strftime("%Y-%m-%d %H:%M:%S.%f UTC"), jd, mjd, gmst)

def magnetic_declination(lat_geod_deg: float, lon_deg: float) -> float:
    geo_mag = GeoMag(coefficients_file="/Users/liamrobinson/Documents/recreational/myquat/data/WMM2025.COF")
    result = geo_mag.calculate(glat=lat_geod_deg, glon=lon_deg, alt=0, time=2025.25)
    return np.deg2rad(result.d)


# WebSocket route to handle continuous geoposition updates
@socketio.on("position_update")
def handle_position_update(data):
    global station
    latitude = float(data.get("latitude_deg"))
    longitude = float(data.get("longitude_deg"))
    frame = data.get('frame')

    station = mr.Station(lat_deg=latitude, lon_deg=longitude)
    if frame == 'enu':
        r = np.array([0.0, 0.0, 0.0])
    elif frame == 'itrf':
        r = station.itrf
    elif frame == 'j2000':
        r = station.j2000_at_dates(mr.now())

    if latitude and longitude:
        # Emit an acknowledgment (optional)
        emit("position_ack", {
            "message": f"Position computed in frame {frame}",
            "r": {"x": r[0], "y": r[1], "z": r[2]},
        })
    else:
        emit("position_ack", {"error": "Invalid position data"})

def get_north_angle(heading_deg, c_p_to_enu):
    # North vector in ENU
    heading_rad = np.deg2rad(heading_deg)
    magnetic_north_enu = np.array([np.sin(heading_rad), np.cos(heading_rad), 0])

    # Get body frame y-axis in ENU
    y_body = c_p_to_enu @ np.array([0, 1, 0])
    
    # Project onto horizontal plane by zeroing z component
    y_body_horizontal = np.array([y_body[0], y_body[1], 0])
    y_body_horizontal = y_body_horizontal / np.linalg.norm(y_body_horizontal)
    
    
    # Get angle between vectors
    angle = np.arccos(np.dot(magnetic_north_enu, y_body_horizontal))
    
    # Determine sign using cross product z component
    cross_z = np.cross(magnetic_north_enu, y_body_horizontal)[2]
    if cross_z < 0:
        angle = -angle
        
    return np.rad2deg(angle)

@socketio.on('orientation_update')
def handle_orientation_update(data):
    if station is None:
        emit("orientation_ack", {"error": "Need position data for orientation solution"})
        return

    frame = data.get("frame")

    # md_rad = magnetic_declination(station.lat_geod_deg, station.lon_deg)

    alpha = float(data.get("alpha_deg"))
    beta = float(data.get("beta_deg"))
    gamma = float(data.get("gamma_deg"))
    compass_heading = float(data.get("compass_heading"))

    c_p_to_enu = mr.r2(-np.deg2rad(gamma)) @ mr.r1(-np.deg2rad(beta)) @ mr.r3(np.deg2rad(alpha))
    n_angle = get_north_angle(compass_heading, c_p_to_enu)

    # c_p_to_enu = mr.r3(np.deg2rad(n_angle)) @ c_p_to_enu

    print(f'{alpha:10.3f}, {beta:10.3f}, {gamma:10.3f}, {compass_heading:10.3f}, {mr.wrap_to_two_pi(compass_heading+alpha):10.3f}, {n_angle:10.3f}')

    c_enu_to_itrf = mr.enu_to_ecef(station.itrf)
    c_j2000_to_itrf = mr.itrf_to_j2000(mr.now())

    if frame == 'enu':
        q = mr.dcm_to_quat(c_p_to_enu).flatten() # takes vectors from p to enu
        # q[:3] = mr.quat_to_rv(q)
        # q[3] = mr.vecnorm(q[:3]).squeeze()
        # q[:3] /= q[3]
    elif frame == 'itrf':
        q = mr.dcm_to_quat(c_enu_in_itrf @ c_p_to_enu).flatten()
    elif frame == 'j2000':
        c = c_j2000_to_itrf @ c_enu_to_itrf @ c_p_to_enu # takes vectors from p to j2000
        q = mr.dcm_to_quat(c).flatten()
        q[:3] = c.T @ np.array([0., 1., 0.])
        q[3] = 0.0
    else:
        raise ValueError(f'Unknown frame name: {frame}, should be enu, itrf, or j2000')

    with open(f'{frame}.quat', 'w') as f:
        f.writelines([f'{x}\n' for x in q])

    if alpha and beta and gamma:
        # Emit an acknowledgment (optional)
        emit("orientation_ack", {
            "message": f"Quaternion computed for {frame}",
            "q": {"x": q[0], "y": q[1], "z": q[2], 'w': q[3]},
        })
    else:
        emit("orientation_ack", {"error": "Invalid position data"})


def main():
    context = ("cert/server.crt", "cert/server.key")  # certificate and key files
    socketio.run(app, port=5001, debug=False, host='0.0.0.0', ssl_context=context)

if __name__ == '__main__':
    main()