🚁 Autonomous Drone Navigation Simulator
A Python-based 3D autonomous drone navigation simulator that combines A* path planning, Kalman-filter state estimation, PID feedback control, simulated distance sensors, and obstacle avoidance in a visual environment.
Simulation only: This is an educational software project, not a tested flight controller for physical drones.

✨ Features
- A pathfinding:* Plans waypoints using a 2D occupancy grid with obstacle clearance.
- 3D flight simulation: Displays drone motion, obstacles, planned waypoints, and flight paths using Matplotlib.
- Kalman filter: Estimates 3D position and velocity from noisy position measurements.
- PID controller: Produces acceleration commands to follow desired motion.
- Virtual distance sensors: Simulates directional obstacle-distance measurements.
- Obstacle avoidance: Adjusts navigation around nearby obstacles and checks movements for collisions.
- Flight phases: Includes takeoff, cruising, landing, and a post-landing hold.
- Live metrics: Shows position, speed, traveled distance, waypoint progress, and minimum clearance.
🛠️ Tech Stack
Technology	Purpose
Python	Simulation logic
NumPy	Vector and matrix calculations
Matplotlib	3D visualization
A*	2D route planning
Kalman filter	State estimation
PID	Feedback control


🚀 Getting Started
Prerequisites
- Python 3.9 or newer
- A graphical desktop session to display the Matplotlib window
Installation
git clone https://github.com/kelvilleshi/autonomous-drone-navigation.git
cd autonomous-drone-navigation
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib


Windows: Activate the environment using .venv\Scripts\activate and use python in place of python3.
Run the simulation
python3 drone_simulation_short.py
A Matplotlib 3D window will open and display the simulated drone mission.
🧠 How It Works
1. Plan: Build a 2D path around rectangular obstacles using A*.
2. Estimate: Use noisy simulated measurements and a Kalman filter to estimate drone state.
3. Navigate: Calculate desired motion toward the current waypoint and respond to virtual distance sensors.
4. Control: Apply PID feedback to produce bounded acceleration and velocity.
5. Simulate: Update position, enforce movement safety checks, and render the flight in 3D.
6. Land: Approach the destination, land, and remain there briefly.

Edit the constants near the top of drone_simulation_short.py to customize:
- START and DESTINATION — flight endpoints
- ROOM_MIN and ROOM_MAX — simulation boundaries
- DT and MAX_STEPS — simulation step and maximum duration
- CRUISE_Z and LAND_Z — cruising and landing altitudes
- SENSOR_RANGE, MAX_SPEED, and MAX_ACCEL — simulated operating limits
- SAFE_RADIUS and PLAN_MARGIN — clearance settings
Modify obstacles() to change the rectangular obstacles.
🔭 Future Improvements
- Full 3D path planning
- Moving obstacles and dynamic replanning
- More realistic drone dynamics and sensor models
- Automated tests and repeatable simulation scenarios
- Flight telemetry export and performance plots
⚠️ Limitations
The current path planner is 2D, even though motion and visualization are 3D. Obstacle avoidance and simulated safety checks do not establish real-world flight safety. This project does not integrate with physical drone hardware.
📄 License
No license has been selected. Add a LICENSE file to specify permitted reuse.
