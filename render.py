import pyvista as pv
import time
import os
import mirage as mr
import numpy as np
import matplotlib.pyplot as plt


def get_north_angle(c_p_to_enu):
    """
    Measures the angular difference between the phone's Y/Z plane
    and the North/Up (horizontal) plane in the ENU frame.

    Parameters:
        c_p_to_enu (np.ndarray): Direction cosine matrix transforming
                                 phone frame to ENU frame.

    Returns:
        float: Angular difference in degrees, positive clockwise when viewed from above.
    """

    def ang(v1, v2) -> float:
        # Compute angle between the two normal vectors
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        # Determine the sign of the angle using the cross product
        cross = np.cross(v1, v2)
        if cross[2] < 0:
            angle = -angle
        return angle

    east_in_enu = np.array([1, 0, 0])
    north_in_enu = np.array([0, 1, 0])
    up_in_enu = np.array([0, 0, 1])

    # Transform the phone's X-axis to ENU frame and project onto the ENU XY plane (no up component)
    phone_x_enu = c_p_to_enu.T @ np.array([1, 0, 0])
    phone_y_enu = c_p_to_enu.T @ np.array([0, 1, 0])
    phone_z_enu = c_p_to_enu.T @ np.array([0, 0, 1])
    phone_x_enu_proj = mr.hat(np.array([*phone_x_enu[:2], 0]))
    phone_y_enu_proj = mr.hat(np.array([*phone_y_enu[:2], 0]))
    phone_z_enu_proj = mr.hat(np.array([*phone_z_enu[:2], 0]))    

    # pl.add_arrows(cent=np.zeros((3,3)), direction=np.array([phone_x_enu_proj, phone_y_enu_proj, phone_z_enu_proj]), render=False, mag=5, name='proj', color='c')
    pl.add_arrows(cent=np.zeros((3,3)), direction=np.array([phone_x_enu, phone_y_enu, phone_z_enu]), render=False, mag=5, name='nproj', color='r')
    pl.add_arrows(cent=np.zeros((3,3)), direction=np.eye(3), render=False, mag=6, name='enu', color='k')

    x_elev = np.arctan2(phone_x_enu[2], np.linalg.norm(phone_x_enu[:2])) # Elevation of the phone x above the local horizontal plane
    y_elev = np.arctan2(phone_y_enu[2], np.linalg.norm(phone_y_enu[:2])) # same for y...
    z_elev = np.arctan2(phone_z_enu[2], np.linalg.norm(phone_z_enu[:2])) # same for z...

    rot_axis = mr.hat(np.cross(phone_z_enu, up_in_enu))
    rv = (np.pi/2-z_elev) * rot_axis
    c_p_to_enu_flat = c_p_to_enu @ mr.rv_to_dcm(rv)

    phone_x_enu_flat = c_p_to_enu_flat.T @ np.array([1, 0, 0])
    phone_y_enu_flat = c_p_to_enu_flat.T @ np.array([0, 1, 0])
    phone_z_enu_flat = c_p_to_enu_flat.T @ np.array([0, 0, 1])

    pl.add_arrows(cent=np.zeros((3,3)), direction=np.array([phone_x_enu_flat, phone_y_enu_flat, phone_z_enu_flat]), render=False, mag=7, name='enu_flat', color='g')

    angle = ang(east_in_enu, phone_x_enu_flat)

    # if z_elev < -np.pi/4:
    #     raise UserWarning("Keep the phone oriented with the screen facing upwards for accurate compass readings")

    return np.rad2deg(angle)


obj = mr.SpaceObject("cube.obj")
obj.v[:, 1] *= 2.0
obj.v[:, 2] /= 4.0
obj = mr.SpaceObject(vertices_and_faces=(obj.v, obj.f))

pl = pv.Plotter()
scalars = np.zeros(12)
scalars[[5, 11]] = 1
pl.add_mesh(obj._mesh, scalars=scalars)
pl.add_mesh(pv.Sphere(radius=10, center=(0.0, 0.0, -13.0)))
pl.camera.position = (0.0, 0.0, 20)
pl.camera.up = (0.0, 1.0, 0)
pl.add_axes()
o_obj = obj._mesh.copy()
pl._on_first_render_request()
pl.render()

inputs = []
outputs = []

while True:
    quat = np.loadtxt("enu.quat")
    if not (len(quat) == 4):
        continue
    # alpha, beta, gamma, compass_heading = quat

    # c_p_to_enu = (
    #     mr.r2(np.deg2rad(gamma))
    #     @ mr.r1(np.deg2rad(beta))
    #     @ mr.r3(np.deg2rad(alpha))
    # )

    # inputs.append(mr.wrap_to_180(compass_heading))
    # outputs.append(mr.wrap_to_180(get_north_angle(c_p_to_enu)))

    # compass_adjustment = mr.wrap_to_360(compass_heading + get_north_angle(c_p_to_enu))
    # c_p_to_enu = c_p_to_enu @ mr.r3(np.deg2rad(-compass_adjustment))

    # print(
    #     f"{compass_heading:6.3f}, {get_north_angle(c_p_to_enu):6.3f} {compass_adjustment:6.3f}"
    # )

    # quat = mr.dcm_to_quat(c_p_to_enu)
    obj.rotate_by_quat(quat)
    pl.update()
    obj._mesh.copy_from(o_obj)
    time.sleep(0.1)
# except KeyboardInterrupt as e:
#     pl.close()
#     plt.scatter(inputs, outputs)
#     plt.show()
