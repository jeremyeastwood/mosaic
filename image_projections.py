import numpy as np
import math
from pyproj import Proj, transform
import pylab
import sys


def deg_to_rad(deg):
    return math.pi*deg/180.


def rad_to_deg(rad):
    return 180*rad/math.pi


def unit_vector(v):
    return v / math.sqrt(float(v.dot(v)))


class Location(object):
    def __init__(self, x=0., y=0., z=0., epsg_code=4326):
        self.x = x
        self.y = y
        self.z = z
        self.epsg_code = epsg_code

    def change_proj(self, out_epsg):
        if self.epsg_code == out_epsg:
            return self.x, self.y
        in_proj = Proj(init='epsg:{}'.format(self.epsg_code))
        out_proj = Proj(init='epsg:{}'.format(out_epsg))
        self.x, self.y = transform(in_proj, out_proj, self.x, self.y)
        self.epsg_code = out_epsg

    @property
    def latlng_json(self):
        x, y = self.latlng_point
        return {'lat': y, 'lng': x}

    @property
    def cartesian_point(self):
        self.change_proj(3857)
        return self.x, self.y

    @property
    def latlng_point(self):
        self.change_proj(4326)
        return self.x, self.y


class Orientation(object):
    def __init__(self, yaw=0., pitch=0., roll=0.):
        self.yaw = 90 - yaw  # correction to heading
        self.pitch = -pitch  # +ve == pitch up
        self.roll = roll  # +ve == roll right
        self.rotation_matrix = self.calc_rotation_matrix()

    def calc_rotation_matrix(self):
        q_roll = self.calc_roll_matrix()
        q_pitch = self.calc_pitch_matrix()
        q_yaw = self.calc_yaw_matrix()
        return np.dot(q_yaw, np.dot(q_pitch, q_roll))

    def calc_roll_matrix(self):
        q = np.array([[1, 0, 0],
                      [0, math.cos(deg_to_rad(self.roll)), -math.sin(deg_to_rad(self.roll))],
                      [0, math.sin(deg_to_rad(self.roll)), math.cos(deg_to_rad(self.roll))]])
        return q

    def calc_pitch_matrix(self):
        q = np.array([[math.cos(deg_to_rad(self.pitch)), 0, math.sin(deg_to_rad(self.pitch))],
                      [0, 1, 0],
                      [-math.sin(deg_to_rad(self.pitch)), 0, math.cos(deg_to_rad(self.pitch))]])
        return q

    def calc_yaw_matrix(self):
        q = np.array([[math.cos(deg_to_rad(self.yaw)), -math.sin(deg_to_rad(self.yaw)), 0],
                      [math.sin(deg_to_rad(self.yaw)), math.cos(deg_to_rad(self.yaw)), 0],
                      [0, 0, 1]])
        return q


class Camera(object):
    def __init__(self, f35=20.):
        self.f35 = f35
        self.sensor_width = 36.0
        self.sensor_height = 24.0
        self.calibration = None  # TODO!
        self.corner_angles = self.calc_corner_angles()
        self.corner_vectors = [self.calc_corner_vector(a) for a in self.corner_angles]

    @property
    def field_of_view(self):
        return rad_to_deg(math.acos(np.dot(self.corner_vectors[0], self.corner_vectors[2])))

    def calc_corner_angles(self):
        center_angle = math.atan(self.sensor_width/self.sensor_height)
        angles = [center_angle]
        angles.append(math.pi - center_angle)
        angles.append(math.pi + center_angle)
        angles.append(2.*math.pi - center_angle)
        return angles

    def calc_corner_vector(self, corner_angle):
        sensor_diameter = math.sqrt(self.sensor_height**2. + self.sensor_width**2.)
        vector = np.array([sensor_diameter*math.cos(corner_angle)/2.,
                           sensor_diameter*math.sin(corner_angle)/2.,
                           -self.f35])
        return unit_vector(vector)


class Image(object):
    def __init__(self, center=None, attitude=None, camera=None):
        self.center = center
        self.attitude = attitude
        self.camera = camera
        self.camera_rays = [self.calculate_cam_ray(v) for v in self.camera.corner_vectors]
        self.intersections = [self.calc_ground_intersection(ray) for ray in self.camera_rays]

    def calculate_cam_ray(self, cam_local_vector):
        return np.dot(self.attitude.rotation_matrix, cam_local_vector)

    def calc_ground_intersection(self, cam_ray):
        cam_x, cam_y = self.center.cartesian_point
        cam_z = self.center.z
        center = np.array([cam_x, cam_y, cam_z])
        ray_z = cam_ray[2]
        if ray_z >= 0:
            print "cam ray will not intersect the ground plane"
            return self.center

        # model ground as flat plane at cam_height from camera
        intersection = center + (-cam_z / ray_z)*cam_ray
        dx = intersection[0] - center[0]
        dy = intersection[1] - center[1]
        s = math.sqrt(dx**2 + dy**2)
        print "intersection point is {}m horizontally from cam center".format(s)
        if s > 5000.:  # can do this at a later stage maybe?
            print "intersection point is too far from cam center (probably a shallow angle) - ignoring"
            return self.center
        return Location(intersection[0], intersection[1], intersection[2], 3857)


def plot_footprints(images):
    pylab.figure()
    for im in images:
        coords = [p.cartesian_point for p in im.intersections]
        coords += [coords[0]]
        x, y = zip(*coords)
        center = im.center.cartesian_point
        pylab.plot(x, y, '-xr')
        pylab.plot(center[0], center[1], 'xk')
    pylab.axes().set_aspect('equal', 'datalim')
    pylab.show()


def load_cam_data(cam_file):
    cam_data = list()
    data = open(cam_file).readlines()[2:-1]
    for line in data:
        cam = line.split('\t')
        try:
            center = Location(*map(float, cam[15:18]))
            attitude = Orientation(*map(float, cam[18:21]))
            cam = Camera(20.0)
            im = Image(center=center, attitude=attitude, camera=cam)
            cam_data.append(im)
        except Exception as e:
            print cam, repr(e)
    return cam_data

if __name__ == '__main__':
    # yaw = 0.
    # pitch = 0.
    # roll = 10.
    # center = Location(-121.444, 45.111, 50.0)
    # attitude = Orientation(yaw=yaw, pitch=pitch, roll=roll)
    # cam = Camera(20.0)
    # print "cam fov: {}deg".format(cam.field_of_view)
    # im = Image(center=center, attitude=attitude, camera=cam)
    # im_nadir = Image(center=center, attitude=Orientation(yaw=yaw), camera=cam)
    # plot_footprints([im, im_nadir])

    cam_file = sys.argv[1]
    cam_data = load_cam_data(cam_file)
    plot_footprints(cam_data)
