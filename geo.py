import math
import datetime

# https://www.sciencedirect.com/topics/engineering/perigee-passage

ticket = 'ticket{papa368346yankee2:GH7VByudpDcHDBMaEAt0SxE3iZ3v_QkahVqj9T0RhNPdTPvW_STlg28vsLW1Pj6Eqw}'
print(ticket)

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

target_a=42164

a = 24732.96738133805
e = 0.7068077979296534
v = 90.38986518838934

# Eccentric Anomaly
E = 2*math.atan(math.sqrt((1 - e) / (1 + e))*math.tan((v*(math.pi/180)) / 2)) # radians or degrees?
# E = 2*math.atan(math.tan(v / 2) / math.sqrt((1 + e) / (1 - e)))  #bad one from stack overflow?

# Distance from earth center
d = math.sqrt(math.pow(8449.401305, 2) + math.pow(9125.794363, 2) + math.pow(-17.461357,2))
print("distance", d)

# Standard gravitational parameter
u = 3.986004418E14

# Velocity calculations
actual_velocity = math.sqrt(math.pow(-1.419072, 2) + math.pow(6.780149, 2) + math.pow(0.002865,2))
calculated_velocity = math.sqrt(u*((2/(d*1000))-(1/(a*1000))))
desired_velocity = math.sqrt(u*((2/(target_a*1000))-(1/(target_a*1000))))
predicted_velocity = math.sqrt(u*((2/(target_a*1000))-(1/(a*1000))))

print("current vel", actual_velocity)
# print("cal vel", calculated_velocity/1000)
print("desired orbit vel", desired_velocity/1000)
print("predicted orbit vel", predicted_velocity/1000)
deltav = desired_velocity/1000 - predicted_velocity/1000
print("deltav", deltav)

# Mean anomaly
Mp = (E - e*math.sin(E))

# time SINCE perigee?
Tp = math.sqrt(math.pow(a*1000, 3) / u)*Mp

# time to apogee?
# Ta = math.sqrt(math.pow(a*1000, 3) / u)*(math.pi - e*math.sin(math.pi))

# orbital period
T = 2*math.pi*math.sqrt(math.pow(a*1000, 3) / u)
transfer_time = math.pi*math.sqrt((math.pow(a*1000+target_a*1000, 3))/(u*8))

print("period", T)
print("perigee passage", Tp)
print("transfer time", transfer_time)


epoch_time = datetime.datetime(2021,6,26,19,20,00)
time_of_perigee = epoch_time - datetime.timedelta(seconds=Tp)
time_of_apogee = time_of_perigee + datetime.timedelta(seconds=transfer_time)
print(epoch_time.strftime("%Y-%m-%d-%H:%M:%S"))
print(time_of_perigee.strftime("%Y-%m-%d-%H:%M:%S"))
print(time_of_apogee.strftime("%Y-%m-%d-%H:%M:%S"))


# current_orbit_perigee
# r1 = (a*(1-e))*1000
# r1 = a*1000
# print("perigee", r1)
# new_orbit_perigee
# r2 = target_a*1000

# this is for the FIRST stage of the transfer, starting from a circular orbit
# deltav = math.sqrt(u/r1)*(math.sqrt((2*r2)/(r1+r2))-1)

# SECOND stage of the transfer, starting from an ellipical orbit
# deltav = math.sqrt(u/r2)*(1-math.sqrt((2*r1)/(r1+r2)))


# print(deltav/1000)
