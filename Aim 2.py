import numpy as np
import math
import matplotlib.pyplot as plt

def rotation_matrix(normal_vector, angle):
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Extract components of the normal vector
    u, v, w = normal_vector
    
    # Compute trigonometric functions
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Compute the rotation matrix elements
    rotation_matrix = np.array([
        [cos_theta + u**2*(1 - cos_theta), u*v*(1 - cos_theta) - w*sin_theta, u*w*(1 - cos_theta) + v*sin_theta],
        [u*v*(1 - cos_theta) + w*sin_theta, cos_theta + v**2*(1 - cos_theta), v*w*(1 - cos_theta) - u*sin_theta],
        [u*w*(1 - cos_theta) - v*sin_theta, v*w*(1 - cos_theta) + u*sin_theta, cos_theta + w**2*(1 - cos_theta)]
    ])
    
    return rotation_matrix

def angle_and_normal(vector1, vector2):
    # Compute the cross product to find the normal vector
    normal_vector = np.cross(vector1, vector2)
    
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Compute the projection of vector1 and vector2 onto the plane perpendicular to the normal vector
    proj_vector1 = vector1 - np.dot(vector1, normal_vector) * normal_vector
    proj_vector2 = vector2 - np.dot(vector2, normal_vector) * normal_vector
    
    # Calculate the cosine of the angle between the projected vectors
    cos_theta = np.dot(proj_vector1, proj_vector2) / (np.linalg.norm(proj_vector1) * np.linalg.norm(proj_vector2))
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    return angle_radians, normal_vector

def points_on_circle(radius, num_points, Z_points):
    # Calculate the angles for each point
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Generate 3D points using polar coordinates
    points = [(radius * np.cos(angle), radius * np.sin(angle), Z_points) for angle in angles]
    return list(zip(points, angles))

    
def add_offsets(data, X_add, Y_add, Z_add):
    modified_data = [(((X + X_add, Y + Y_add, Z + Z_add), angle)) for ((X, Y, Z), angle) in data]
    return modified_data

def rotate_data(data, rotation_matrix):
    modified_data = [((list(np.dot(rotation_matrix, np.array([X, Y, Z]))), angle)) for ((X, Y, Z), angle) in data]
    return modified_data
    

def closest2points(data1, data2):
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # Initialize minimum distance and corresponding angles
    min_distance = float('inf')
    min_angle1, min_angle2 = None, None
    min_angle1_data1, min_angle2_data1 = None, None
    min_angle1_data2, min_angle2_data2 = None, None
    
    # Iterate over all pairs of points to find the closest ones
    for (x1, y1, _), angle1 in data1:
        for (x2, y2, _), angle2 in data2:
            dist = distance((x1, y1), (x2, y2))
            if dist < min_distance:
                min_distance = dist
                min_angle1_data1, min_angle2_data1 = angle1, angle2
    
    min_distance = float('inf')
    
    for (x1, y1, _), angle1 in data1:
        for (x2, y2, _), angle2 in data2:
            dist = distance((x1, y1), (x2, y2))
            if dist < min_distance:
                if (angle1, angle2) != (min_angle1_data1, min_angle2_data1):
                    min_distance = dist
                    min_angle1_data2, min_angle2_data2 = angle1, angle2
    
    return (min_angle1_data1, min_angle2_data1), (min_angle1_data2, min_angle2_data2)

def plot_data(data1, data2):
    """
    Plots the (X, Y) coordinates from two datasets.
    
    Parameters:
        data1: First dataset of the form (((X, Y, Z), associated_angle), ...).
        data2: Second dataset of the form (((X, Y, Z), associated_angle), ...).
    """
    # Extract X and Y coordinates from data1 and data2
    x1 = [point[0][0] for point in data1]
    y1 = [point[0][1] for point in data1]
    x2 = [point[0][0] for point in data2]
    y2 = [point[0][1] for point in data2]
    
    # Plot data1 and data2
    plt.scatter(x1, y1, color='blue', label='Data 1')
    plt.scatter(x2, y2, color='red', label='Data 2')
    
    # Set plot labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of X, Y Coordinates')
    plt.legend()
    
    # Show plot
    plt.show()
    
# Example usage:
radius1 = 2
num_points = 100
radius2 = 1.5
Z_dist = 10
circle1 = points_on_circle(radius1, num_points, 0)
circle2 = points_on_circle(radius2, num_points, Z_dist)

# Example usage:
vector1 = np.array([0, 0, Z_dist])
Target= np.array([3, 3, 30])

print("circle1")
print("(X, Y, Z), associated_angle")
for stuff in circle1:
    print(stuff)
print("circle2")
print("(X, Y, Z), associated_angle")
for stuff2 in circle2:
    print(stuff2)
    
print("here is the target position:")
print(Target)

angle, normal_vector = angle_and_normal(vector1, Target)
print("Angle between vector1 and vector2 about the normal vector:", angle)
print("Normal vector:", normal_vector)

R = rotation_matrix(normal_vector, angle)
print("Rotation Matrix:")
print(R)

modified_circle1 = add_offsets(circle1, Target[0], Target[1], Target[2])
modified_circle2 = add_offsets(circle2, Target[0], Target[1], Target[2])
print("mod_circle1")
print("(X, Y, Z), associated_angle")
for stuffs in modified_circle1:
    print(stuffs)
print("mod_circle2")
print("(X, Y, Z), associated_angle")
for stuffs in modified_circle2:
    print(stuffs)
    
Rot_circ1 = rotate_data(modified_circle1, R)    
Rot_circ2 = rotate_data(modified_circle2, R)   

print("Rot_circle1")
print("(X, Y, Z), associated_angle")
for stuffs in Rot_circ1:
    print(stuffs)
print("Rot_circle2")
print("(X, Y, Z), associated_angle")
for stuffs in Rot_circ2:
    print(stuffs)
    
    
angleset1, angleset2 = closest2points(Rot_circ1,Rot_circ2)

print("here are the two angle sets")
print(angleset1)
print(angleset2)
'''
plot_data(Rot_circ1, Rot_circ2)

cam1point1 = (radius1*np.cos(angleset1[0]),radius1*np.sin(angleset1[0]),Z_dist)
cam1point2 = (radius1*np.cos(angleset1[1]),radius1*np.sin(angleset1[1]),0)
cam2point1 = (radius2*np.cos(angleset2[0]),radius2*np.sin(angleset2[0]),Z_dist)
cam2point2 = (radius2*np.cos(angleset2[1]),radius2*np.sin(angleset2[1]),0)
'''


all3Dpoints = circle1+ circle2 + [((Target[0],Target[1],Target[2]),0)]
#[Target, cam1point1, cam1point2, cam2point1, cam2point2]
print(all3Dpoints)

X = [point[0][0] for point in all3Dpoints]
Y = [point[0][1] for point in all3Dpoints]
Z = [point[0][2] for point in all3Dpoints]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, color= 'red',label='Points')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
