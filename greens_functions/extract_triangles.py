import numpy as np
from rsqsim_api.fault.segment import RsqSimSegment
import os.path
import os

cutde_dir = "."
for dip in [30, 40, 50]:
    fault = RsqSimSegment.from_stl(os.path.join(cutde_dir, f"hundalee{dip}_remeshed2km.stl"))
    tris = fault.patch_vertices
    np.save(os.path.join(cutde_dir, f"hundalee{dip}_remeshed2km.npy"), tris)
    if not os.path.exists("../inversion/plotting"):
        os.mkdir("../inversion/plotting")
    np.save(f"../inversion/plotting/hundalee{dip}_remeshed2km_tris.npy", fault.triangles)
    np.save(f"../inversion/plotting/hundalee{dip}_remeshed2km_verts.npy", fault.vertices)