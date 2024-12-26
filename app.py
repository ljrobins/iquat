from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import mirage as mr
import numpy as np
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

@socketio.on('orientation_update')
def handle_orientation_update(data):
    if station is None:
        emit("orientation_ack", {"error": "Need position data for orientation solution"})
        return

    print(data)
    frame = data.get("frame")

    alpha = np.deg2rad(float(data.get("alpha_deg")))
    beta = np.deg2rad(float(data.get("beta_deg")))
    gamma = np.deg2rad(float(data.get("gamma_deg")))

    c_user_in_enu = mr.r3(alpha) @ mr.r1(beta) @ mr.r2(gamma)
    # c_user_in_enu = mr.ea_to_dcm((2,1,3), gamma, beta, alpha)
    # print(c, c_user_in_enu)
    c_enu_in_itrf = mr.enu_to_ecef(station.itrf)
    c_j2000_in_itrf = mr.itrf_to_j2000(mr.now())

    if frame == 'enu':
        q = mr.dcm_to_quat(c_user_in_enu).flatten()
        # q[:3] = mr.quat_to_rv(q)
        # q[3] = mr.vecnorm(q[:3]).squeeze()
        # q[:3] /= q[3]
    elif frame == 'itrf':
        q = mr.dcm_to_quat(c_enu_in_itrf @ c_user_in_enu).flatten()
    elif frame == 'j2000':
        q = mr.dcm_to_quat(c_j2000_in_itrf @ c_enu_in_itrf @ c_user_in_enu).flatten()
    else:
        raise ValueError(f'Unknown frame name: {frame}, should be enu, itrf, or j2000')

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
    socketio.run(app, port=5001, debug=True, host='0.0.0.0', ssl_context=context)

if __name__ == '__main__':
    main()