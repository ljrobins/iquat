import pyvista as pv
import time
import os
import mirage as mr
import numpy as np

obj = mr.SpaceObject('cube.obj')
obj.v[:,1] *= 2.0
obj.v[:,2] /= 4.0
obj = mr.SpaceObject(vertices_and_faces=(obj.v, obj.f))

pl = pv.Plotter()
scalars = np.zeros(12)
scalars[[5, 11]] = 1
pl.add_mesh(obj._mesh, scalars=scalars)
pl.add_mesh(pv.Sphere(radius=10, center=(0., 0., -13.)))
pl.camera.position = (0., 0., 20)
pl.camera.up = (0., 1., 0)
pl.add_axes()
o_obj = obj._mesh.copy()
pl._on_first_render_request()
pl.render()

old_quat = None
while True:
    quat = np.loadtxt('enu.quat')
    if not (len(quat) == 4):
        continue

    if old_quat is None:
        old_quat = quat

    if ~np.allclose(old_quat, quat):
        print(quat)
        obj.rotate_by_quat(quat)
        pl.update()
        obj._mesh.copy_from(o_obj)
    old_quat = quat

pl.close()

filename = "enu.quat"
