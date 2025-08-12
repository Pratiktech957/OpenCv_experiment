import cv2
import numpy as np
import math

# Projection parameters
def project_3d_to_2d(point, angle_x, angle_y, distance=5):
    """Rotate point in 3D and project to 2D."""
    x, y, z = point

    # Rotation around X axis
    y2 = y * math.cos(angle_x) - z * math.sin(angle_x)
    z2 = y * math.sin(angle_x) + z * math.cos(angle_x)

    # Rotation around Y axis
    x3 = x * math.cos(angle_y) + z2 * math.sin(angle_y)
    z3 = -x * math.sin(angle_y) + z2 * math.cos(angle_y)

    # Perspective projection
    factor = distance / (distance + z3)
    u = int(256 + x3 * factor * 100)
    v = int(256 - y2 * factor * 100)
    return (u, v)

# Stickman joints in 3D space
points_3d = [
    (0, 1.8, 0),   # Head top
    (0, 1.5, 0),   # Neck
    (-0.4, 1.5, 0),  # Left shoulder
    (-0.4, 1.0, 0),  # Left elbow
    (-0.4, 0.5, 0),  # Left hand
    (0.4, 1.5, 0),   # Right shoulder
    (0.4, 1.0, 0),   # Right elbow
    (0.4, 0.5, 0),   # Right hand
    (0, 1.0, 0),     # Chest
    (-0.2, 0.5, 0),  # Left hip
    (-0.2, 0.0, 0),  # Left knee
    (-0.2, -0.5, 0), # Left foot
    (0.2, 0.5, 0),   # Right hip
    (0.2, 0.0, 0),   # Right knee
    (0.2, -0.5, 0),  # Right foot
]

# Body connections (pairs of point indices)
bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14)
]

# Animation loop
angle = 0
while True:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    angle += 0.02

    # Project all points
    projected = [project_3d_to_2d(p, angle_x=0.5, angle_y=angle) for p in points_3d]

    # Draw bones
    for a, b in bones:
        cv2.line(img, projected[a], projected[b], (0, 255, 0), 2)

    # Draw joints
    for pt in projected:
        cv2.circle(img, pt, 4, (0, 0, 255), -1)

    cv2.imshow("3D Stickman (Projected)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
