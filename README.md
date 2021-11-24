## Hack-a-Sat 2
### Cotten-Eye GEO Challenge

Problem: Given the initial state vector for a satellite, calcuate the time, direction, and magnitude for an orbital burn to place the satellite in a Geosynchronous orbit.  

Solution: Need to calcuate the second burn of a two-burn Hohmann transfer orbit. Use the initial state vector to propagate the satellite to apogee. The second burn should be performed at the time of apogee and should be in the same direction (velocity vector) of the motion of the satellite. The magnitude of the burn can be calcuated using the vis-viva equation.
