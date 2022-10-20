import numpy as np

omega = 365.*2*np.pi
dt = 0.001

dtheta = np.pi/2
sin_theta = np.sin(dtheta)
cos_theta = np.sqrt(1 - sin_theta**2)

# or if less than threshold make zero??

# print(sin_theta)
# print(cos_theta)

sx = 0.
sy = 0.
sz = 1.

ix = 1.
iy = 0.
iz = 0.

jx = 0.
jy = 1.
jz = 0.

kx = 0.
ky = 0.
kz = 1.

r = np.zeros((3,3))
r[0,0] = cos_theta + (sx * sx * (1-cos_theta))
r[0,1] = (sx * sy * (1-cos_theta)) - (sz * sin_theta)
r[0,2] = (sx * sz * (1-cos_theta)) + (sy * sin_theta)
r[1,0] = (sx * sy * (1-cos_theta)) + (sz * sin_theta)
r[1,1] = cos_theta + (sy * sy * (1-cos_theta))
r[1,2] = (sz * sy * (1-cos_theta)) - (sx * sin_theta)
r[2,0] = (sx * sz * (1-cos_theta)) - (sy * sin_theta)
r[2,1] = (sz * sy * (1-cos_theta)) + (sx * sin_theta)
r[2,2] = cos_theta + (sz * sz * (1-cos_theta))

# print(r[0,0]**2 + r[0,1]**2 + r[0,2]**2)
# print(r[1,0]**2 + r[1,1]**2 + r[1,2]**2)
# print(r[2,0]**2 + r[2,1]**2 + r[2,2]**2)
# print(r[0,0]**2 + r[1,0]**2 + r[2,0]**2)
# print(r[0,1]**2 + r[1,1]**2 + r[2,1]**2)
# print(r[0,2]**2 + r[1,2]**2 + r[2,2]**2)

# print(r[0,0]*r[0,1] + r[1,0]*r[1,1] + r[2,0]*r[2,1])
# print(r[0,0]*r[0,2] + r[1,0]*r[1,2] + r[2,0]*r[2,2])
# print(r[0,2]*r[0,1] + r[1,2]*r[1,1] + r[2,2]*r[2,1])
# print(r[1,0]*r[0,0] + r[1,1]*r[0,1] + r[1,2]*r[0,2])
# print(r[2,0]*r[0,0] + r[2,1]*r[0,1] + r[2,2]*r[0,2])
# print(r[1,0]*r[2,0] + r[1,1]*r[2,1] + r[1,2]*r[2,2])

ix_ = ix * r[0,0] + iy * r[0,1] + iz * r[0,2]
iy_ = ix * r[1,0] + iy * r[1,1] + iz * r[1,2]
iz_ = ix * r[2,0] + iy * r[2,1] + iz * r[2,2]
jx_ = jx * r[0,0] + jy * r[0,1] + jz * r[0,2]
jy_ = jx * r[1,0] + jy * r[1,1] + jz * r[1,2]
jz_ = jx * r[2,0] + jy * r[2,1] + jz * r[2,2]
kx_ = kx * r[0,0] + ky * r[0,1] + kz * r[0,2]
ky_ = kx * r[1,0] + ky * r[1,1] + kz * r[1,2]
kz_ = kx * r[2,0] + ky * r[2,1] + kz * r[2,2]

print(ix_)
print(iy_)
print(iz_)
print(jx_)
print(jy_)
print(jz_)
print(kx_)
print(ky_)
print(kz_)