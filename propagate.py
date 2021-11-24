import math
import datetime
from sgp4.api import Satrec, WGS84, jday, SGP4_ERRORS
from sgp4 import exporter
from sgp4.conveniences import jday_datetime
import numpy as np
import socket
import json

# Standard gravitational parameter for Earth
u = 3.9860044188E14


class OrbitalElements:
    
    def __init__(self):
        self.semimajor_axis = 0
        self.eccentricity = 0
        self.inclination = 0
        self.omega = 0
        self.argp = 0
        self.nu = 0
        self.M = 0
        self.T = 0
        self.mm = 0

    # Fundamentals of Astrodynamics and Applications, by Vallado, 2007.
    # https://space.stackexchange.com/questions/1904/how-to-programmatically-calculate-orbital-elements-using-position-velocity-vecto
    def calculate(self, pos, vel):
        pos = np.multiply(pos, 1000)
        vel = np.multiply(vel, 1000)
        angular_momentum = np.cross(pos, vel)
        node_vector = np.cross([0, 0, 1], angular_momentum)
        term1 = np.multiply((math.pow(np.linalg.norm(vel), 2) - u / np.linalg.norm(pos)), pos)
        term2 = np.multiply(np.dot(pos, vel), vel)
        eccentricity_vector = (term1 - term2) / u
        eccentricity = np.linalg.norm(eccentricity_vector)

        energy = (math.pow(np.linalg.norm(vel), 2) / 2) - (u / np.linalg.norm(pos))

        # wtf is eps?
        # if abs(e - 1.0) > eps
        semimajor_axis = -u / (2 * energy)
        p = semimajor_axis * (1 - math.pow(eccentricity, 2))
        # else
        #    p = mag(h) ^ 2 / mu
        #    a = inf

        inclination = math.acos(angular_momentum[2] / np.linalg.norm(angular_momentum))
        omega = math.acos(node_vector[0] / np.linalg.norm(node_vector))

        if node_vector[1] < 0:
            omega = 2 * math.pi - omega

        argp = math.acos(np.dot(node_vector, eccentricity_vector) / (np.linalg.norm(node_vector) * eccentricity))

        if eccentricity_vector[2] < 0:
            argp = 2 * math.pi - argp

        nu = math.acos(np.dot(eccentricity_vector, pos) / (eccentricity * np.linalg.norm(pos)))

        if np.dot(pos, vel) < 0:
            nu = 2 * math.pi - nu

        # Eccentric Anomaly
        E = 2 * math.atan(math.sqrt((1 - eccentricity) / (1 + eccentricity)) * math.tan(nu / 2))
        # Mean anomaly
        M = E - (eccentricity * math.sin(E))
        # Orbital period
        T = 2 * math.pi * math.sqrt(math.pow(semimajor_axis, 3) / u)
        # Mean motion revs/day
        mm = 86400 / T
        mm_r = mm * (1 / 1440) * (2 * math.pi)  # convert to radians/minute

        print("           Orbital Elements Summary           ")
        print("==============================================")
        print("Semimajor axis, a (km): ", semimajor_axis / 1000)
        print("Eccentricity, e: ", eccentricity)
        print("Inclination, i (deg): ", inclination * (180.0 / math.pi))
        print("Right ascension of the ascending node, Ω (deg): ", omega * (180.0 / math.pi))
        print("Argument of perigee, ω (deg): ", argp * (180.0 / math.pi))
        print("True anomaly, υ (deg): ", nu * (180.0 / math.pi))
        print("Mean anomaly, M (deg): ", M * (180.0 / math.pi))
        print("Orbital period, T (sec): ", T)
        print("Mean motion, n (revs/day): ", mm)
        print("Mean motion (radians/min): ", mm_r)
        print("==============================================")
        print()

        self.semimajor_axis = semimajor_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.omega = omega
        self.argp = argp
        self.nu = nu
        self.M = M
        self.T = T
        self.mm = mm_r


def get_hohmann_transfer_dv1(apogee_distance, perigee_distance):
    deltav = math.sqrt(u / (perigee_distance * 1000)) * (
                math.sqrt((2 * apogee_distance * 1000) / ((apogee_distance * 1000) + (perigee_distance * 1000))) - 1)
    return deltav


def get_hohmann_transfer_dv2(apogee_distance, perigee_distance):
    deltav = math.sqrt(u / (apogee_distance * 1000)) * (
                1 - math.sqrt((2 * perigee_distance * 1000) / ((apogee_distance * 1000) + (perigee_distance * 1000))))
    return deltav


def vis_viva(radius, semi_major_axis):
    return math.sqrt(u * ((2 / (radius * 1000)) - (1 / (semi_major_axis * 1000))))


def read_input(filename):
    data = {}
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def save_tle(filename, line1, line2):
    with open(filename, 'w') as fp:
        print(line1, file=fp)
        print(line2, file=fp)


def submit_answer(ticket, time, deltav):
    HOST = 'visual-sun.satellitesabove.me'
    PORT = 5014

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        data = s.recv(1024)
        print(data.decode())
        s.sendall(ticket.encode())
        s.sendall(b'\n')
        while True:
            data = s.recv(1024)
            if not data:
                break
            else:
                print(data.decode())
            if 'Time: 2021' in data.decode():
                s.sendall(time.encode())
                s.sendall(b'\n')
            if 'v_x' in data.decode():
                s.sendall(str(deltav[0]).encode())
            if 'v_y' in data.decode():
                s.sendall(str(deltav[1]).encode())
            if 'v_z' in data.decode():
                s.sendall(str(deltav[2]).encode())


ticket = 'ticket{papa368346yankee2:GH7VByudpDcHDBMaEAt0SxE3iZ3v_QkahVqj9T0RhNPdTPvW_STlg28vsLW1Pj6Eqw}'
# print(ticket)

# https://www.sciencedirect.com/topics/engineering/perigee-passage

# Semimajor axis, a (km): 24732.96738133805
# Eccentricity, e: 0.7068077979296534
# Inclination, i (deg): 0.11790360842507447
# Right ascension of the ascending node, Ω (deg): 90.22650379956278
# Argument of perigee, ω (deg): 226.58754885495142
# True anomaly, υ (deg): 90.38986518838934

# Pos (km):   [8449.401305, 9125.794363, -17.461357]
# Vel (km/s): [-1.419072, 6.780149, 0.002865]
# Time:       2021-06-26-19:20:00.000-UTC

# a=42164+/-10km, e<0.001, i<1deg

input = read_input('inputs.json')
original_pos = input['position']
original_vel = input['velocity']

print("           Pre-Maneuver Ephemeris             ")
print("==============================================")
print("Position: ", original_pos)
print("Velocity vector: ", original_vel)
print("==============================================")
print()

elements = OrbitalElements()
elements.calculate(original_pos, original_vel)

julian_epoch = datetime.datetime(1949, 12, 31, 0, 0, 0, tzinfo=datetime.timezone.utc)
sat_epoch = datetime.datetime.strptime(input['epoch'], '%Y-%m-%d-%H:%M:%S.000-UTC')
sat_epoch = sat_epoch.replace(tzinfo=datetime.timezone.utc)
delta = sat_epoch - julian_epoch
julian = delta.days + delta.seconds/86400

sat = Satrec()
ret = sat.sgp4init(
    WGS84,            # gravity model
    'i',              # 'a' = old AFSPC mode, 'i' = improved mode
    50000,            # satnum: Satellite number
    julian,           # epoch: days since 1949 December 31 00:00 UT
    0.0,              # 2.8098e-05 # bstar: drag coefficient (1/earth radii)
    6.969196665e-13,  # ndot (NOT USED): ballistic coefficient (revs/day)
    0.0,              # nddot (NOT USED): mean motion 2nd derivative (revs/day^3)
    elements.eccentricity,   # ecco: eccentricity
    elements.argp,           # argpo: argument of perigee (radians)
    elements.inclination,    # inclo: inclination (radians)
    elements.M,              # mo: mean anomaly (radians)
    elements.mm,             # no_kozai: mean motion (radians/minute)
    elements.omega           # nodeo: right ascension of ascending node (radians)
)

line1, line2 = exporter.export_tle(sat)
print("           Pre-Maneuver TLE                   ")
print("==============================================")
print(line1)
print(line2)
print("==============================================")
print()

# found by brute force, not exact calculation
# propagated the satellite until the max distance from earth and the min velocity magnitude was found
time_array = []
prev_time = sat_epoch
for i in range(0, int(elements.T)):
    time_array.append(prev_time)
    prev_time = prev_time + datetime.timedelta(0, 1)  # days, seconds, then other fields.

jd_array = []
fr_array = []
for t in time_array:
    jd, fr = jday_datetime(t)
    jd_array.append(jd)
    fr_array.append(fr)

error, r, v = sat.sgp4_array(np.array(jd_array), np.array(fr_array))
apogee_distance = 0
perigee_distance = 100000
time_index = 0
time_index_perigee = 0
for i in range(0, len(r)):
    d = np.linalg.norm(r[i])
    if d > apogee_distance:
        apogee_distance = d
        time_index = i
    if d < perigee_distance:
        perigee_distance = d
        time_index_perigee = i

# Velocity magnitude
vv = np.linalg.norm(v[time_index])

print("           Propagation Result                 ")
print("==============================================")
print("Apoapsis distance (km): {} at {}".format(apogee_distance, time_array[time_index]))
print("Apoapsis velocity (km/s): ", vv)
print("Periapsis distance (km): {} at {}".format(perigee_distance, time_array[time_index_perigee]))
print("==============================================")
print()

# velocity unit vector
v_hat = v[time_index] / vv
# print(v_hat)

print("      Predicted Post-Maneuver Elements        ")
print("==============================================")
target_eccentricity = input['target_eccentricity'] - 0.00001
print("Target Eccentricity, e: ", target_eccentricity)
target_sma = (apogee_distance/(1+target_eccentricity))
print("Target Semimajor axis, a (km): ", target_sma)
target_velocity = vis_viva(apogee_distance, target_sma)
print("==============================================")
print()

print("           Maneuver Parameters                ")
print("==============================================")
final_dv = target_velocity/1000 - vv
print("Required delta v, dv (km/s): ", final_dv)
final_dv_vector = final_dv*v_hat
print("Delta v vector: ", final_dv_vector)
print("Maneuver time: ", time_array[time_index])
print("==============================================")
print()

print("           Post-Maneuver Ephemeris            ")
print("==============================================")
print("Position: ", r[time_index])
new_velocity_vector = np.add(v[time_index], final_dv_vector)
print("New velocity vector: ", new_velocity_vector)
print("==============================================")
print()
# submit_answer(ticket, tt, final_dv_vector)

newElements = OrbitalElements()
newElements.calculate(r[time_index], new_velocity_vector)

delta = time_array[time_index] - julian_epoch
new_julian = delta.days + delta.seconds/86400

newSat = Satrec()
ret = newSat.sgp4init(
    WGS84,            # gravity model
    'i',              # 'a' = old AFSPC mode, 'i' = improved mode
    50001,            # satnum: Satellite number
    new_julian,       # epoch: days since 1949 December 31 00:00 UT
    0.0,              # 2.8098e-05 # bstar: drag coefficient (1/earth radii)
    6.969196665e-13,  # ndot (NOT USED): ballistic coefficient (revs/day)
    0.0,              # nddot (NOT USED): mean motion 2nd derivative (revs/day^3)
    newElements.eccentricity,   # ecco: eccentricity
    newElements.argp,           # argpo: argument of perigee (radians)
    newElements.inclination,    # inclo: inclination (radians)
    newElements.M,              # mo: mean anomaly (radians)
    newElements.mm,             # no_kozai: mean motion (radians/minute)
    newElements.omega           # nodeo: right ascension of ascending node (radians)
)

print("           Post-Maneuver TLE                  ")
print("==============================================")
line1, line2 = exporter.export_tle(newSat)
print(line1)
print(line2)
print("==============================================")
print()

print("           Post-Maneuver Analysis             ")
print("==============================================")
if (newElements.semimajor_axis/1000 < float(input['target_semi_major_axis']+input['target_sma_error']) and
    newElements.semimajor_axis/1000 > float(input['target_semi_major_axis']-input['target_sma_error'])):
    print("Post-Maneuver semi-major axis within defined parameters")
else:
    print("Post-Maneuver semi-major axis outside of defined parameters!")

if newElements.eccentricity < input['target_eccentricity']:
    print("Post-Maneuver eccentricity within defined parameters")
else:
    print("Post-Maneuver eccentricity outside of defined parameters!")

if newElements.inclination < input['target_inclination']:
    print("Post-Maneuver inclination within defined parameters")
else:
    print("Post-Maneuver inclination outside of defined parameters!")
print("==============================================")
print()