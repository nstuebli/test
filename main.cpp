import numpy as np

class BearingOnlyEKF:
    """
    Extended Kalman Filter for target tracking in Cartesian coordinates [x, y, vx, vy].
    Supports dynamic measurement modes: 
      - 'B'    : [Bearing]
      - 'BR'   : [Bearing, Range]
      - 'BRCS' : [Bearing, Range, Speed, Course]
    """
    def __init__(self, dt: float, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_dict: dict):
        """
        Args:
            dt: Time step (seconds)
            x0: Initial state [x, y, vx, vy] (4x1 or 1D array)
            P0: Initial covariance matrix (4x4)
            Q: Process noise covariance matrix (4x4)
            R_dict: Dictionary mapping measurement types to covariance matrices:
                    {'B': (1,1), 'BR': (2,2), 'BRCS': (4,4)}
        """
        self.dt = dt
        self.x = np.array(x0, dtype=float).reshape(4, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_dict = {key: np.array(val, dtype=float) for key, val in R_dict.items()}

        # Constant Velocity State Transition Matrix (4x4)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wraps angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def h(self, x: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B') -> np.ndarray:
        """Observation model mapping state vector to measurement space (returns mx1 array)."""
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos.flatten()

        dx = px - x_o
        dy = py - y_o

        bearing = np.arctan2(dy, dx)
        rng = np.hypot(dx, dy)

        if measurement_type == 'B':
            return np.array([[bearing]])
        
        elif measurement_type == 'BR':
            return np.array([[bearing], [rng]])
        
        elif measurement_type == 'BRCS':
            speed = np.hypot(vx, vy)
            course = np.arctan2(vy, vx)
            return np.array([[bearing], [rng], [speed], [course]])
        
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def H_jacobian(self, x: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B') -> np.ndarray:
        """Computes observation Jacobian matrix H (mx4)."""
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos.flatten()
        
        dx = px - x_o
        dy = py - y_o

        r2 = dx*dx + dy*dy
        r2_safe = max(r2, 1e-6)
        r_safe = np.sqrt(r2_safe)

        # Bearing & Range Jacobians (1x4 each)
        Hb = np.array([[-dy / r2_safe, dx / r2_safe, 0.0, 0.0]])
        Hr = np.array([[dx / r_safe, dy / r_safe, 0.0, 0.0]])

        if measurement_type == 'B':
            return Hb
        elif measurement_type == 'BR':
            return np.vstack([Hb, Hr])
        elif measurement_type == 'BRCS':
            v2 = vx*vx + vy*vy
            v2_safe = max(v2, 1e-6)
            v_safe = np.sqrt(v2_safe)

            # Speed & Course Jacobians (1x4 each)
            Hs = np.array([[0.0, 0.0, vx / v_safe, vy / v_safe]])
            Hc = np.array([[0.0, 0.0, -vy / v2_safe, vx / v2_safe]])

            return np.vstack([Hb, Hr, Hs, Hc])
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def predict(self):
        """Propagates state and covariance forward by dt."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_value: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B'):
        """Applies Kalman update for a given measurement."""
        z = np.array(measurement_value, dtype=float).reshape(-1, 1)
        z_pred = self.h(self.x, own_pos, measurement_type)
        
        # Innovation vector (mx1)
        y = z - z_pred

        # Angle wrapping on bearing (index 0) and course (index 3 if BRCS)
        y[0, 0] = self._wrap_angle(y[0, 0])
        if measurement_type == 'BRCS':
            y[3, 0] = self._wrap_angle(y[3, 0])

        H = self.H_jacobian(self.x, own_pos, measurement_type)
        R = self.R_dict[measurement_type]

        # Innovation covariance (mxm)
        S = H @ self.P @ H.T + R

        # Kalman Gain (4xm) using linear solver for numerical stability
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # State & Covariance Update
        self.x = self.x + K @ y
        I = np.eye(4)
        
        # Joseph Form update for guaranteed numerical symmetry/positive definiteness
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T


class GeoProjection:
    """Converts between Lat/Lon (deg) and Local ENU Cartesian coordinates (meters)."""
    R_EARTH = 6371000.0  # Earth radius in meters

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = np.radians(lat0)
        self.lon0 = np.radians(lon0)
        self.cos_lat0 = np.cos(self.lat0)

    def geodetic_to_enu(self, lat: float, lon: float) -> np.ndarray:
        """Converts (lat, lon) in degrees to [East, North] position in meters."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        dlat = lat_rad - self.lat0
        dlon = lon_rad - self.lon0

        x_east = dlon * self.cos_lat0 * self.R_EARTH
        y_north = dlat * self.R_EARTH
        return np.array([x_east, y_north])

    def enu_to_geodetic(self, x_east: float, y_north: float) -> tuple[float, float]:
        """Converts local [East, North] in meters back to (lat, lon) in degrees."""
        dlat = y_north / self.R_EARTH
        dlon = x_east / (self.R_EARTH * self.cos_lat0)

        lat = np.degrees(self.lat0 + dlat)
        lon = np.degrees(self.lon0 + dlon)
        return lat, lon


# 1. Time step (seconds)
dt = 1.0

# 2. Initial State Estimate x0: [x, y, vx, vy] in meters and m/s
x0 = np.array([1000.0, 2000.0, 10.0, -5.0]) 

# 3. Initial Covariance P0 (4x4): Uncertainty in x0
# e.g., position uncertainty std dev = 50 m (var = 2500), velocity std dev = 5 m/s (var = 25)
P0 = np.diag([50.0**2, 50.0**2, 5.0**2, 5.0**2])

# 4. Process Noise Covariance Q (4x4): Model uncertainty per time step
# Represents unmodeled target accelerations (e.g., std dev = 0.1 m/s^2)
q_var = 0.1**2
Q = np.diag([0.5 * dt**2 * q_var, 0.5 * dt**2 * q_var, dt * q_var, dt * q_var])

# 5. Measurement Noise Dictionary R_dict
sigma_bearing = np.radians(1.0)  # 1 degree bearing noise converted to radians
sigma_range   = 10.0            # 10 meters range noise
sigma_speed   = 0.5             # 0.5 m/s speed noise
sigma_course  = np.radians(2.0)  # 2 degrees course noise converted to radians

R_dict = {
    'B': np.array([
        [sigma_bearing**2]
    ]),
    
    'BR': np.diag([
        sigma_bearing**2, 
        sigma_range**2
    ]),
    
    'BRCS': np.diag([
        sigma_bearing**2, 
        sigma_range**2, 
        sigma_speed**2, 
        sigma_course**2
    ])
}

# Create filter instance
ekf = BearingOnlyEKF(dt=dt, x0=x0, P0=P0, Q=Q, R_dict=R_dict)




# =====================================================================
# 2. SIMULATION SETUP
# =====================================================================
dt = 1.0
total_time = 300  # seconds
steps = int(total_time / dt)

# Maneuver Timing
maneuver_start = 100.0  # seconds
maneuver_end = 130.0    # seconds

# Target Ground Truth (Constant Velocity)
target_pos = np.array([3000.0, 4000.0])  # [East, North] in meters
target_vel = np.array([-5.0, 2.0])        # [vx, vy] in m/s

# Own-ship Setup
own_pos = np.array([0.0, 0.0])            # Starts at origin
own_speed = 8.0                            # 8 m/s (~15.5 knots)
heading_deg = 0.0                          # Initial heading East (0 deg)
turn_rate = 90.0 / (maneuver_end - maneuver_start)  # 3 deg/s left turn to North

# Measurement Noise
sigma_bearing_deg = 1.0  # 1 degree noise
R_dict_deg = {'B': [[sigma_bearing_deg**2]]}

# EKF Initial Guess (Deliberately wrong range/velocity to test convergence)
x0_guess = [2000.0, 2500.0, 0.0, 0.0]     # Estimated pos offset by ~2 km
P0 = np.diag([1500.0**2, 1500.0**2, 10.0**2, 10.0**2])
Q = np.diag([0.01, 0.01, 0.001, 0.001])

ekf = BearingOnlyEKF(dt=dt, x0=x0_guess, P0=P0, Q=Q, R_dict_deg=R_dict_deg)

# Data logging arrays
history_target_true = []
history_own_ship = []
history_ekf_est = []
history_pos_error = []
history_is_maneuvering = []

# =====================================================================
# 3. MAIN SIMULATION LOOP
# =====================================================================
np.random.seed(42)

for i in range(steps):
    t = i * dt
    is_maneuvering = maneuver_start <= t <= maneuver_end

    # 1. Update Own-ship Kinetics
    if is_maneuvering:
        heading_deg += turn_rate * dt  # Turning toward North (90 deg)
    
    heading_rad = np.radians(heading_deg)
    own_vel = np.array([own_speed * np.cos(heading_rad), own_speed * np.sin(heading_rad)])
    own_pos += own_vel * dt

    # 2. Update Target Ground Truth
    target_pos += target_vel * dt

    # 3. EKF Predict Step (Always performed)
    ekf.predict()

    # 4. EKF Update Step (PAUSED DURING MANEUVER)
    if not is_maneuvering:
        # Generate noisy bearing measurement
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        true_bearing_deg = np.degrees(np.arctan2(dy, dx))
        noisy_bearing_deg = true_bearing_deg + np.random.normal(0, sigma_bearing_deg)

        ekf.update(measurement_value_deg=[noisy_bearing_deg], own_pos=own_pos, measurement_type='B')

    # Logging
    history_target_true.append(target_pos.copy())
    history_own_ship.append(own_pos.copy())
    history_ekf_est.append(ekf.x.flatten().copy())
    history_is_maneuvering.append(is_maneuvering)

    # Position estimation error
    est_pos = ekf.x[:2].flatten()
    history_pos_error.append(np.linalg.norm(target_pos - est_pos))

# =====================================================================
# 4. VISUALIZATION
# =====================================================================
target_true = np.array(history_target_true)
own_ship = np.array(history_own_ship)
ekf_est = np.array(history_ekf_est)
time_vec = np.arange(steps) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: 2D Spatial Trajectory
ax1.plot(own_ship[:, 0], own_ship[:, 1], 'b-', label='Own-ship Path')
maneuver_idx = np.where(history_is_maneuvering)[0]
ax1.plot(own_ship[maneuver_idx, 0], own_ship[maneuver_idx, 1], 'r-', linewidth=3, label='Own-ship Maneuver (Updates Paused)')
ax1.plot(target_true[:, 0], target_true[:, 1], 'g--', label='Target True Path')
ax1.plot(ekf_est[:, 0], ekf_est[:, 1], 'm:', label='EKF Estimated Path')
ax1.scatter(own_ship[0, 0], own_ship[0, 1], color='blue', marker='o', label='Own-ship Start')
ax1.scatter(target_true[0, 0], target_true[0, 1], color='green', marker='o', label='Target Start')
ax1.set_title('2D Target Tracking Trajectory')
ax1.set_xlabel('East [m]')
ax1.set_ylabel('North [m]')
ax1.grid(True)
ax1.legend()

# Plot 2: Position Estimation Error Over Time
ax2.plot(time_vec, history_pos_error, 'k-', label='Position Error (m)')
ax2.axvspan(maneuver_start, maneuver_end, color='red', alpha=0.2, label='Maneuver Window')
ax2.set_title('Target Estimation Error vs Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position Error [m]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

=====================================
    import numpy as np

class BearingOnlyEKF:
    """
    Extended Kalman Filter for target tracking in Cartesian coordinates [x, y, vx, vy].
    Supports dynamic measurement modes: 
      - 'B'    : [Bearing]
      - 'BR'   : [Bearing, Range]
      - 'BRCS' : [Bearing, Range, Speed, Course]
    """
    def __init__(self, dt: float, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_dict: dict):
        """
        Args:
            dt: Time step (seconds)
            x0: Initial state [x, y, vx, vy] (4x1 or 1D array)
            P0: Initial covariance matrix (4x4)
            Q: Process noise covariance matrix (4x4)
            R_dict: Dictionary mapping measurement types to covariance matrices:
                    {'B': (1,1), 'BR': (2,2), 'BRCS': (4,4)}
        """
        self.dt = dt
        self.x = np.array(x0, dtype=float).reshape(4, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_dict = {key: np.array(val, dtype=float) for key, val in R_dict.items()}

        # Constant Velocity State Transition Matrix (4x4)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wraps angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def h(self, x: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B') -> np.ndarray:
        """Observation model mapping state vector to measurement space (returns mx1 array)."""
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos.flatten()

        dx = px - x_o
        dy = py - y_o

        bearing = np.arctan2(dy, dx)
        rng = np.hypot(dx, dy)

        if measurement_type == 'B':
            return np.array([[bearing]])
        
        elif measurement_type == 'BR':
            return np.array([[bearing], [rng]])
        
        elif measurement_type == 'BRCS':
            speed = np.hypot(vx, vy)
            course = np.arctan2(vy, vx)
            return np.array([[bearing], [rng], [course], [speed]])
        
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def H_jacobian(self, x: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B') -> np.ndarray:
        """Computes observation Jacobian matrix H (mx4)."""
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos.flatten()
        
        dx = px - x_o
        dy = py - y_o

        r2 = dx*dx + dy*dy
        r2_safe = max(r2, 1e-6)
        r_safe = np.sqrt(r2_safe)

        # Bearing and Range Jacobians (1x4)
        Hb = np.array([[-dy / r2_safe, dx / r2_safe, 0.0, 0.0]])
        Hr = np.array([[dx / r_safe, dy / r_safe, 0.0, 0.0]])

        if measurement_type == 'B':
            return Hb
        elif measurement_type == 'BR':
            return np.vstack([Hb, Hr])
        elif measurement_type == 'BRCS':
            v2 = vx*vx + vy*vy
            v2_safe = max(v2, 1e-6)
            v_safe = np.sqrt(v2_safe)

            # Speed and Course Jacobians (1x4)
            Hc = np.array([[0.0, 0.0, -vy / v2_safe, vx / v2_safe]])
            Hs = np.array([[0.0, 0.0, vx / v_safe, vy / v_safe]])

            return np.vstack([Hb, Hr, Hc, Hs])
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def predict(self):
        """Propagates state and covariance forward by dt."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_value: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B'):
        """Applies Kalman update for a given measurement."""
        z = np.array(measurement_value, dtype=float).reshape(-1, 1)
        z_pred = self.h(self.x, own_pos, measurement_type)
        
        # Innovation vector (mx1)
        y = z - z_pred

        # Angle wrapping on bearing (index 0) and course
        y[0, 0] = self._wrap_angle(y[0, 0])
        if measurement_type == 'BRCS':
            y[3, 0] = self._wrap_angle(y[3, 0])

        H = self.H_jacobian(self.x, own_pos, measurement_type)
        R = self.R_dict[measurement_type]

        # Innovation covariance (mxm)
        S = H @ self.P @ H.T + R

        # Kalman Gain (4xm)
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # State and Covariance Update
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

==================================
import numpy as np
import math
import matplotlib.pyplot as plt
import ekf
import dataimporter

time, meas = dataimporter.load_sensortrack('track4_sensor13_CAS_BDT.log')
_, os = dataimporter.load_ownship('track4_objectData.log')
_, tgt = dataimporter.load_ownship('track4_objectData.log')

# Lat/Lon converter


dt = 1.0
total_time = 15 * 60  # seconds
steps = int(total_time / dt)

# Maneuver Timing
maneuver_start = 120.0  # seconds
maneuver_end = 150.0    # seconds

# Target Ground Truth (Constant Velocity)
target_pos = np.array([-3000.0, -4000.0])  # [East, North] in meters
target_vel = np.array([-5.0, 2.0])        # [vx, vy] in m/s

# Own-ship Setup
own_pos = np.array([0.0, 0.0])            # Starts at origin
own_speed = 2.0                            # 8 m/s (~15.5 knots)
heading_deg = 0.0                          # Initial heading East (0 deg)
turn_rate = 90.0 / (maneuver_end - maneuver_start)  # 3 deg/s left turn to North

# Measurement Noise
sigma_B = np.radians(0.2)
sigma_R = 5.0
sigma_C = np.radians(1.0)
sigma_S = 0.2
R_dict = {
    'B': np.array([
        [sigma_B**2]
    ]),
    
    'BR': np.diag([
        sigma_B**2, 
        sigma_R**2
    ]),
    
    'BRCS': np.diag([
        sigma_B**2, 
        sigma_R**2, 
        sigma_C**2, 
        sigma_S**2
    ])
}

# EKF Initial Guess (Deliberately wrong range/velocity to test convergence)
x0_guess = [2000.0, 2500.0, -4.0, 1.5]     # Estimated pos offset by ~2 km
P0 = np.diag([1500.0**2, 1500.0**2, 10.0**2, 10.0**2])
Q = np.diag([0.01, 0.01, 0.001, 0.001])

ekf = ekf.BearingOnlyEKF(dt=dt, x0=x0_guess, P0=P0, Q=Q, R_dict=R_dict)

# Data logging arrays
history_target_true = []
history_own_ship = []
history_ekf_est = []
history_pos_error = []
history_is_maneuvering = []

# =====================================================================

np.random.seed(42)

for i in range(steps):
    t = i * dt
    is_maneuvering = maneuver_start <= t <= maneuver_end

    # 1. Update Own-ship Kinetics
    if is_maneuvering:
        heading_deg += turn_rate * dt  # Turning toward North (90 deg)
    
    heading_rad = np.radians(heading_deg)
    own_vel = np.array([own_speed * np.cos(heading_rad), own_speed * np.sin(heading_rad)])
    own_pos += own_vel * dt

    # 2. Update Target Ground Truth
    target_pos += target_vel * dt

    # 3. EKF Predict Step (Always performed)
    ekf.predict()

    # 4. EKF Update Step (PAUSED DURING MANEUVER)
    if not is_maneuvering:
        # Generate noisy bearing measurement
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        true_bearing_deg = np.degrees(np.arctan2(dy, dx))
        noisy_bearing_deg = true_bearing_deg + np.random.normal(0, sigma_B)

        ekf.update(measurement_value=[noisy_bearing_deg], own_pos=own_pos, measurement_type='B')

    # Logging
    history_target_true.append(target_pos.copy())
    history_own_ship.append(own_pos.copy())
    history_ekf_est.append(ekf.x.flatten().copy())
    history_is_maneuvering.append(is_maneuvering)

    # Position estimation error
    est_pos = ekf.x[:2].flatten()
    history_pos_error.append(np.linalg.norm(target_pos - est_pos))

# =====================================================================

target_true = np.array(history_target_true)
own_ship = np.array(history_own_ship)
ekf_est = np.array(history_ekf_est)
time_vec = np.arange(steps) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: 2D Spatial Trajectory
ax1.plot(own_ship[:, 0], own_ship[:, 1], 'b-', label='Own-ship Path')
ax1.plot(target_true[:, 0], target_true[:, 1], 'g--', label='Target True Path')
ax1.plot(ekf_est[:, 0], ekf_est[:, 1], 'm:', label='EKF Estimated Path')
ax1.set_title('2D Target Tracking Trajectory')
ax1.set_xlabel('East [m]')
ax1.set_ylabel('North [m]')
ax1.grid(True)
ax1.legend()

# Plot 2: Position Estimation Error Over Time
ax2.plot(time_vec, history_pos_error, 'k-', label='Position Error (m)')
ax2.set_title('Target Estimation Error vs Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position Error [m]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
