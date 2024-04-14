import tf_conversions as tfc
import numpy as np
from IPython import embed

# Camera body wrt RGB link
pq_c_wrt_r = ((-0.032, 0, 0), (-0.5, -0.5, 0.5, 0.5))
T_c_wrt_r = tfc.toMatrix(tfc.fromTf(pq_c_wrt_r))

# TCP wrt RGB link (from hand-eye calibration)
pq_r_wrt_e = ( (-0.070853, 0.0405379, 0.0795092) , (0.713858, 0.699918, 0.0166051, 0.0157051))
T_r_wrt_e = tfc.toMatrix(tfc.fromTf(pq_r_wrt_e))

T_c_wrt_e = T_r_wrt_e @ T_c_wrt_r
pq_c_wrt_e = tfc.toTf(tfc.fromMatrix(T_c_wrt_e))


# R_new = np.array([[0, 1, 0],
#                   [0, 0, -1],
#                   [-1, 0, 0]])

# R_new_extended = np.eye(4)
# R_new_extended[:3, :3] = R_new

# T_c_wrt_e = R_new_extended @ T_c_wrt_e  

# Convert the final transformation matrix back to pose format if needed
pq_c_wrt_e = tfc.toTf(tfc.fromMatrix(T_c_wrt_e))

print()


embed()

