import math
import datetime
from sgp4.api import Satrec, WGS84, jday, SGP4_ERRORS
import numpy as np
import socket


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

# https://www.sciencedirect.com/topics/engineering/perigee-passage

ticket = 'ticket{papa368346yankee2:GH7VByudpDcHDBMaEAt0SxE3iZ3v_QkahVqj9T0RhNPdTPvW_STlg28vsLW1Pj6Eqw}'
print(ticket)

#Semimajor axis, a (km): 24732.96738133805
#Eccentricity, e: 0.7068077979296534
#Inclination, i (deg): 0.11790360842507447
#Right ascension of the ascending node, Ω (deg): 90.22650379956278
#Argument of perigee, ω (deg): 226.58754885495142
#True anomaly, υ (deg): 90.38986518838934

#Pos (km):   [8449.401305, 9125.794363, -17.461357]
#Vel (km/s): [-1.419072, 6.780149, 0.002865]
#Time:       2021-06-26-19:20:00.000-UTC

# a=42164+/-10km, e<0.001, i<1deg

#Standard gravitational parameter for Earth
u = 3.9860044188E14

#orbital parameters
a = 24732.96738133805
e =   0.7068077979296534
v =  90.38986518838934
w = 226.58754885495142
i =   0.11790360842507447
omega = 90.22650379956278

#conversions
a_m = a*1000
w_rad = w*(math.pi/180.0)
i_rad = i*(math.pi/180.0)
v_rad = v*(math.pi/180.0)
omega_rad = omega*(math.pi/180.0)

#Eccentric Anomaly
E = 2*math.atan(math.sqrt((1 - e) / (1 + e))*math.tan(v_rad / 2))
#Mean anomaly
M = E - (e*math.sin(E))
#Orbital period
T = 2*math.pi*math.sqrt(math.pow(a_m, 3) / u)
#Mean motion revs/day
n = 86400/T
n = n*(1/1440)*(2*math.pi)   #convert to radians/minute

julian_epoch = datetime.datetime(1949, 12, 31, 0, 0, 0)
sat_epoch = datetime.datetime(2021, 6, 26, 19, 20, 0)
delta = sat_epoch - julian_epoch
julian = delta.days + delta.seconds/86400

sat = Satrec()
ret = sat.sgp4init(
    WGS84,           # gravity model
    'i',             # 'a' = old AFSPC mode, 'i' = improved mode
    50000,           # satnum: Satellite number
    julian,          # epoch: days since 1949 December 31 00:00 UT
    0.0, #2.8098e-05,      # bstar: drag coefficient (1/earth radii)
    6.969196665e-13, # ndot (NOT USED): ballistic coefficient (revs/day)
    0.0,             # nddot (NOT USED): mean motion 2nd derivative (revs/day^3)
    e,               # ecco: eccentricity
    w_rad,           # argpo: argument of perigee (radians)
    i_rad,           # inclo: inclination (radians)
    M,               # mo: mean anomaly (radians)
    n,               # no_kozai: mean motion (radians/minute)
    omega_rad,       # nodeo: right ascension of ascending node (radians)
)

# 100km higher than planned, for some reason
target_a=42174*1000
#target_a=42200*1000
min_vv = 10

#found by trial-and-error, not exact calculation
#propagated the satellite until the max distance from earth and the min velocity magnitude was found
#2021-06-27-00:12:37.418 apogee time

#for i in np.arange(37.417,37.418,0.001):
for i in range(11,14):
    jd, fr = jday(2021, 6, 27, 0, i, 37.418)
    error, r, v = sat.sgp4(jd, fr)

    if error:
        print(SGP4_ERRORS[error])
        exit()
    
    # Distance from earth center
    d = math.sqrt(math.pow(r[0], 2) + math.pow(r[1], 2) + math.pow(r[2],2))

    tt = '2021-06-27-00:{}:37.418-UTC'.format(i)
    print(tt)
    print(r)
    print(v)
    print("distance", d)

    # Velocity magnitude
    vv = np.linalg.norm(v)
    print("velocity", vv)

    if vv < min_vv:
        min_vv = vv
        ttv = tt

    # unit vector
    v_hat = v / vv
    #print(v_hat)

    desired_sma = (42173.75)*1000
    desired_r = 42205
    desired_velocity = math.sqrt(u*((2/(d*1000))-(1/desired_sma)))

    final_dv = desired_velocity/1000 - vv
    print("delta v", final_dv)
    final_dv_vector = final_dv*v_hat
    print("new velocity vector", np.add(np.asarray(v), final_dv_vector))
    print(final_dv_vector)
    print()
    #submit_answer(ticket, tt, final_dv_vector)

