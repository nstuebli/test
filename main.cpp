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
    Extended Kalman Filter operating strictly in standard SI units (meters, m/s, radians).
    """
    def __init__(self, dt: float, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_dict: dict):
        self.dt = dt
        self.x = np.array(x0, dtype=float).reshape(4, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_dict = {key: np.array(val, dtype=float) for key, val in R_dict.items()}

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """Wraps angle in radians to [-pi, pi]."""
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    def h(self, x: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B') -> np.ndarray:
        """Returns predictions in standard SI units (radians, meters, m/s)."""
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
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos.flatten()
        dx, dy = px - x_o, py - y_o

        r2_safe = max(dx*dx + dy*dy, 1e-6)
        r_safe = np.sqrt(r2_safe)

        Hb = np.array([[-dy / r2_safe, dx / r2_safe, 0.0, 0.0]])
        Hr = np.array([[dx / r_safe, dy / r_safe, 0.0, 0.0]])

        if measurement_type == 'B':
            return Hb
        elif measurement_type == 'BR':
            return np.vstack([Hb, Hr])
        elif measurement_type == 'BRCS':
            v2_safe = max(vx*vx + vy*vy, 1e-6)
            v_safe = np.sqrt(v2_safe)

            Hc = np.array([[0.0, 0.0, -vy / v2_safe, vx / v2_safe]])
            Hs = np.array([[0.0, 0.0, vx / v_safe, vy / v_safe]])

            return np.vstack([Hb, Hr, Hc, Hs])
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_value: np.ndarray, own_pos: np.ndarray, measurement_type: str = 'B'):
        z = np.array(measurement_value, dtype=float).reshape(-1, 1)
        z_pred = self.h(self.x, own_pos, measurement_type)
        
        y = z - z_pred

        # Wrap bearing (index 0) and course (index 2 for BRCS)
        y[0, 0] = self._wrap_angle(y[0, 0])
        if measurement_type == 'BRCS':
            y[2, 0] = self._wrap_angle(y[2, 0])  # Fixed: index 2 is course

        H = self.H_jacobian(self.x, own_pos, measurement_type)
        R = self.R_dict[measurement_type]

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

==================================
import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
total_time = 15 * 60
steps = int(total_time / dt)

maneuver_start = 120.0
maneuver_end = 150.0

target_pos = np.array([-3000.0, -4000.0])
target_vel = np.array([-5.0, 2.0])

own_pos = np.array([0.0, 0.0])
own_speed = 8.0
heading_deg = 0.0
turn_rate = 90.0 / (maneuver_end - maneuver_start)

# Noise Parameters (Standard SI: Radians and meters)
sigma_B_deg = 0.2
sigma_B_rad = np.radians(sigma_B_deg)

R_dict = {'B': np.array([[sigma_B_rad**2]])}

# Initial Guess: Pointed along the correct initial line of bearing (~233 degrees)
# Range guess is deliberately wrong (2000m vs 5000m actual) to test range convergence
init_bearing = np.arctan2(target_pos[1], target_pos[0])
init_range_guess = 2000.0

x0_guess = [
    init_range_guess * np.cos(init_bearing),
    init_range_guess * np.sin(init_bearing),
    -4.0, 1.5
]

P0 = np.diag([1500.0**2, 1500.0**2, 10.0**2, 10.0**2])
Q = np.diag([0.01, 0.01, 0.001, 0.001])

ekf = BearingOnlyEKF(dt=dt, x0=x0_guess, P0=P0, Q=Q, R_dict=R_dict)

history_target_true = []
history_own_ship = []
history_ekf_est = []
history_pos_error = []

np.random.seed(42)

for i in range(steps):
    t = i * dt
    is_maneuvering = maneuver_start <= t <= maneuver_end

    if is_maneuvering:
        heading_deg += turn_rate * dt
    
    heading_rad = np.radians(heading_deg)
    own_vel = np.array([own_speed * np.cos(heading_rad), own_speed * np.sin(heading_rad)])
    own_pos += own_vel * dt
    target_pos += target_vel * dt

    ekf.predict()

    if not is_maneuvering:
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        
        # Calculate true bearing in RADIANS and add RADIAN noise
        true_bearing_rad = np.arctan2(dy, dx)
        noisy_bearing_rad = true_bearing_rad + np.random.normal(0, sigma_B_rad)

        ekf.update(measurement_value=[noisy_bearing_rad], own_pos=own_pos, measurement_type='B')

    history_target_true.append(target_pos.copy())
    history_own_ship.append(own_pos.copy())
    history_ekf_est.append(ekf.x.flatten().copy())
    history_pos_error.append(np.linalg.norm(target_pos - ekf.x[:2].flatten()))

# Plotting
target_true = np.array(history_target_true)
own_ship = np.array(history_own_ship)
ekf_est = np.array(history_ekf_est)
time_vec = np.arange(steps) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(own_ship[:, 0], own_ship[:, 1], 'b-', label='Own-ship Path')
ax1.plot(target_true[:, 0], target_true[:, 1], 'g--', label='Target True Path')
ax1.plot(ekf_est[:, 0], ekf_est[:, 1], 'm:', label='EKF Estimated Path')
ax1.set_title('2D Target Tracking Trajectory')
ax1.set_xlabel('East [m]')
ax1.set_ylabel('North [m]')
ax1.grid(True)
ax1.legend()

ax2.plot(time_vec, history_pos_error, 'k-', label='Position Error (m)')
ax2.axvspan(maneuver_start, maneuver_end, color='red', alpha=0.2, label='Maneuver Window')
ax2.set_title('Target Estimation Error vs Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position Error [m]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

    ================================
import numpy as np

class MPCEKF:
    """
    EKF in Modified Polar Coordinates (MPC) supporting dynamic measurement modes:
      - 'B'    : [Bearing]
      - 'BR'   : [Bearing, Range]
      - 'BRCS' : [Bearing, Range, Course, Speed]
    
    State vector y = [beta, beta_dot, rho, zeta]^T:
      - beta     : Bearing angle [rad]
      - beta_dot : Bearing rate [rad/s]
      - rho      : Normalized range rate (r_dot / r) [1/s]
      - zeta     : Inverse range (1 / r) [1/m]
    """
    def __init__(self, y0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_dict: dict):
        self.y = np.array(y0, dtype=float).reshape(4, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_dict = {key: np.array(val, dtype=float) for key, val in R_dict.items()}

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """Wraps angle in radians to [-pi, pi]."""
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def cartesian_to_mpc(target_pos: np.ndarray, target_vel: np.ndarray, 
                         own_pos: np.ndarray, own_vel: np.ndarray) -> np.ndarray:
        """Converts Cartesian positions/velocities to an MPC state vector."""
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        dvx = target_vel[0] - own_vel[0]
        dvy = target_vel[1] - own_vel[1]

        r2 = max(dx*dx + dy*dy, 1e-6)
        r = np.sqrt(r2)

        beta = np.arctan2(dy, dx)
        beta_dot = (dx * dvy - dy * dvx) / r2
        rho = (dx * dvx + dy * dvy) / r2
        zeta = 1.0 / r

        return np.array([beta, beta_dot, rho, zeta]).reshape(4, 1)

    def to_cartesian(self, own_pos: np.ndarray, own_vel: np.ndarray) -> np.ndarray:
        """Converts current MPC state to Cartesian target state [x, y, vx, vy]."""
        beta, beta_dot, rho, zeta = self.y.flatten()
        r = 1.0 / max(abs(zeta), 1e-6)

        dx = r * np.cos(beta)
        dy = r * np.sin(beta)
        
        dvx = r * (rho * np.cos(beta) - beta_dot * np.sin(beta))
        dvy = r * (rho * np.sin(beta) + beta_dot * np.cos(beta))

        return np.array([own_pos[0] + dx, own_pos[1] + dy, own_vel[0] + dvx, own_vel[1] + dvy])

    def _get_target_velocity(self, y: np.ndarray, own_vel: np.ndarray):
        """Computes Cartesian target velocity [v_tx, v_ty] and its jacobian w.r.t y."""
        beta, beta_dot, rho, zeta = y.flatten()
        v_ox, v_oy = own_vel.flatten()

        zeta_safe = max(abs(zeta), 1e-6)
        r = 1.0 / zeta_safe

        dvx = r * (rho * np.cos(beta) - beta_dot * np.sin(beta))
        dvy = r * (rho * np.sin(beta) + beta_dot * np.cos(beta))

        v_tx = v_ox + dvx
        v_ty = v_oy + dvy

        # Derivatives of [v_tx, v_ty] w.r.t y = [beta, beta_dot, rho, zeta]
        dv_dy = np.array([
            [-dvy + v_oy,  -r * np.sin(beta),  r * np.cos(beta),  -dvx / zeta_safe],
            [ dvx - v_ox,   r * np.cos(beta),  r * np.sin(beta),  -dvy / zeta_safe]
        ])

        return np.array([v_tx, v_ty]), dv_dy

    def h(self, y: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B') -> np.ndarray:
        """Observation function for 'B', 'BR', and 'BRCS'."""
        beta, _, _, zeta = y.flatten()

        if measurement_type == 'B':
            return np.array([[beta]])
        
        rng = 1.0 / max(abs(zeta), 1e-6)
        if measurement_type == 'BR':
            return np.array([[beta], [rng]])
        
        if measurement_type == 'BRCS':
            if own_vel is None:
                raise ValueError("own_vel is required to compute Course and Speed in MPC.")
            (v_tx, v_ty), _ = self._get_target_velocity(y, own_vel)
            course = np.arctan2(v_ty, v_tx)
            speed = np.hypot(v_tx, v_ty)
            return np.array([[beta], [rng], [course], [speed]])

        raise ValueError(f"Unknown measurement type: {measurement_type}")

    def H_jacobian(self, y: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B') -> np.ndarray:
        """Computes measurement Jacobian matrix H."""
        _, _, _, zeta = y.flatten()
        zeta_safe = max(abs(zeta), 1e-6)

        Hb = np.array([[1.0, 0.0, 0.0, 0.0]])
        Hr = np.array([[0.0, 0.0, 0.0, -1.0 / (zeta_safe**2)]])

        if measurement_type == 'B':
            return Hb
        elif measurement_type == 'BR':
            return np.vstack([Hb, Hr])
        elif measurement_type == 'BRCS':
            if own_vel is None:
                raise ValueError("own_vel is required to compute H for Course and Speed.")
            
            (v_tx, v_ty), dv_dy = self._get_target_velocity(y, own_vel)
            v2 = max(v_tx**2 + v_ty**2, 1e-6)
            v_norm = np.sqrt(v2)

            # Course Jacobian: dC/dy
            dC_dv = np.array([[-v_ty / v2, v_tx / v2]])
            Hc = dC_dv @ dv_dy

            # Speed Jacobian: dS/dy
            dS_dv = np.array([[v_tx / v_norm, v_ty / v_norm]])
            Hs = dS_dv @ dv_dy

            return np.vstack([Hb, Hr, Hc, Hs])

        raise ValueError(f"Unknown measurement type: {measurement_type}")

    def _dynamics(self, y: np.ndarray, own_accel: np.ndarray) -> np.ndarray:
        """Continuous state derivative dy/dt = f(y, a_o)."""
        beta, beta_dot, rho, zeta = y.flatten()
        ax_o, ay_o = own_accel.flatten()

        a_or = -ax_o * np.cos(beta) - ay_o * np.sin(beta)
        a_ob =  ax_o * np.sin(beta) - ay_o * np.cos(beta)

        d_beta     = beta_dot
        d_beta_dot = -2.0 * beta_dot * rho + zeta * a_ob
        d_rho      = beta_dot**2 - rho**2 + zeta * a_or
        d_zeta     = -rho * zeta

        return np.array([[d_beta], [d_beta_dot], [d_rho], [d_zeta]])

    def _system_jacobian(self, y: np.ndarray, own_accel: np.ndarray) -> np.ndarray:
        """Continuous system Jacobian Matrix A = df/dy."""
        beta, beta_dot, rho, zeta = y.flatten()
        ax_o, ay_o = own_accel.flatten()

        a_or = -ax_o * np.cos(beta) - ay_o * np.sin(beta)
        a_ob =  ax_o * np.sin(beta) - ay_o * np.cos(beta)

        A = np.zeros((4, 4))
        A[0, 1] = 1.0

        A[1, 0] = -zeta * a_or
        A[1, 1] = -2.0 * rho
        A[1, 2] = -2.0 * beta_dot
        A[1, 3] = a_ob

        A[2, 0] = zeta * a_ob
        A[2, 1] = 2.0 * beta_dot
        A[2, 2] = -2.0 * rho
        A[2, 3] = a_or

        A[3, 2] = -zeta
        A[3, 3] = -rho

        return A

    def predict(self, dt: float, own_accel: np.ndarray = np.array([0.0, 0.0])):
        """Propagates state (RK4) and covariance."""
        k1 = self._dynamics(self.y, own_accel)
        k2 = self._dynamics(self.y + 0.5 * dt * k1, own_accel)
        k3 = self._dynamics(self.y + 0.5 * dt * k2, own_accel)
        k4 = self._dynamics(self.y + dt * k3, own_accel)
        
        self.y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.y[0, 0] = self._wrap_angle(self.y[0, 0])

        A = self._system_jacobian(self.y, own_accel)
        F = np.eye(4) + A * dt + 0.5 * (A @ A) * (dt**2)

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement_value: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B'):
        """Applies Kalman update for 'B', 'BR', or 'BRCS'."""
        z = np.array(measurement_value, dtype=float).reshape(-1, 1)
        z_pred = self.h(self.y, own_vel=own_vel, measurement_type=measurement_type)

        y_res = z - z_pred
        
        # Angle wrapping for bearing (index 0) and course (index 2 in BRCS)
        y_res[0, 0] = self._wrap_angle(y_res[0, 0])
        if measurement_type == 'BRCS':
            y_res[2, 0] = self._wrap_angle(y_res[2, 0])

        H = self.H_jacobian(self.y, own_vel=own_vel, measurement_type=measurement_type)
        R = self.R_dict[measurement_type]

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.y = self.y + K @ y_res
        self.y[0, 0] = self._wrap_angle(self.y[0, 0])

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

==============================================
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. MPC EKF CLASS IMPLEMENTATION
# =====================================================================
class MPCEKF:
    def __init__(self, y0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_dict: dict):
        self.y = np.array(y0, dtype=float).reshape(4, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R_dict = {key: np.array(val, dtype=float) for key, val in R_dict.items()}

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def cartesian_to_mpc(target_pos: np.ndarray, target_vel: np.ndarray, 
                         own_pos: np.ndarray, own_vel: np.ndarray) -> np.ndarray:
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        dvx = target_vel[0] - own_vel[0]
        dvy = target_vel[1] - own_vel[1]

        r2 = max(dx*dx + dy*dy, 1e-6)
        r = np.sqrt(r2)

        beta = np.arctan2(dy, dx)
        beta_dot = (dx * dvy - dy * dvx) / r2
        rho = (dx * dvx + dy * dvy) / r2
        zeta = 1.0 / r

        return np.array([beta, beta_dot, rho, zeta]).reshape(4, 1)

    def to_cartesian(self, own_pos: np.ndarray, own_vel: np.ndarray) -> np.ndarray:
        beta, beta_dot, rho, zeta = self.y.flatten()
        r = 1.0 / max(abs(zeta), 1e-6)

        dx = r * np.cos(beta)
        dy = r * np.sin(beta)
        
        dvx = r * (rho * np.cos(beta) - beta_dot * np.sin(beta))
        dvy = r * (rho * np.sin(beta) + beta_dot * np.cos(beta))

        return np.array([own_pos[0] + dx, own_pos[1] + dy, own_vel[0] + dvx, own_vel[1] + dvy])

    def _get_target_velocity(self, y: np.ndarray, own_vel: np.ndarray):
        beta, beta_dot, rho, zeta = y.flatten()
        v_ox, v_oy = own_vel.flatten()

        zeta_safe = max(abs(zeta), 1e-6)
        r = 1.0 / zeta_safe

        dvx = r * (rho * np.cos(beta) - beta_dot * np.sin(beta))
        dvy = r * (rho * np.sin(beta) + beta_dot * np.cos(beta))

        v_tx = v_ox + dvx
        v_ty = v_oy + dvy

        dv_dy = np.array([
            [-dvy + v_oy,  -r * np.sin(beta),  r * np.cos(beta),  -dvx / zeta_safe],
            [ dvx - v_ox,   r * np.cos(beta),  r * np.sin(beta),  -dvy / zeta_safe]
        ])

        return np.array([v_tx, v_ty]), dv_dy

    def h(self, y: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B') -> np.ndarray:
        beta, _, _, zeta = y.flatten()

        if measurement_type == 'B':
            return np.array([[beta]])
        
        rng = 1.0 / max(abs(zeta), 1e-6)
        if measurement_type == 'BR':
            return np.array([[beta], [rng]])
        
        if measurement_type == 'BRCS':
            if own_vel is None:
                raise ValueError("own_vel is required for BRCS.")
            (v_tx, v_ty), _ = self._get_target_velocity(y, own_vel)
            course = np.arctan2(v_ty, v_tx)
            speed = np.hypot(v_tx, v_ty)
            return np.array([[beta], [rng], [course], [speed]])

        raise ValueError(f"Unknown measurement type: {measurement_type}")

    def H_jacobian(self, y: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B') -> np.ndarray:
        _, _, _, zeta = y.flatten()
        zeta_safe = max(abs(zeta), 1e-6)

        Hb = np.array([[1.0, 0.0, 0.0, 0.0]])
        Hr = np.array([[0.0, 0.0, 0.0, -1.0 / (zeta_safe**2)]])

        if measurement_type == 'B':
            return Hb
        elif measurement_type == 'BR':
            return np.vstack([Hb, Hr])
        elif measurement_type == 'BRCS':
            if own_vel is None:
                raise ValueError("own_vel is required for BRCS.")
            
            (v_tx, v_ty), dv_dy = self._get_target_velocity(y, own_vel)
            v2 = max(v_tx**2 + v_ty**2, 1e-6)
            v_norm = np.sqrt(v2)

            dC_dv = np.array([[-v_ty / v2, v_tx / v2]])
            Hc = dC_dv @ dv_dy

            dS_dv = np.array([[v_tx / v_norm, v_ty / v_norm]])
            Hs = dS_dv @ dv_dy

            return np.vstack([Hb, Hr, Hc, Hs])

        raise ValueError(f"Unknown measurement type: {measurement_type}")

    def _dynamics(self, y: np.ndarray, own_accel: np.ndarray) -> np.ndarray:
        beta, beta_dot, rho, zeta = y.flatten()
        ax_o, ay_o = own_accel.flatten()

        a_or = -ax_o * np.cos(beta) - ay_o * np.sin(beta)
        a_ob =  ax_o * np.sin(beta) - ay_o * np.cos(beta)

        d_beta     = beta_dot
        d_beta_dot = -2.0 * beta_dot * rho + zeta * a_ob
        d_rho      = beta_dot**2 - rho**2 + zeta * a_or
        d_zeta     = -rho * zeta

        return np.array([[d_beta], [d_beta_dot], [d_rho], [d_zeta]])

    def _system_jacobian(self, y: np.ndarray, own_accel: np.ndarray) -> np.ndarray:
        beta, beta_dot, rho, zeta = y.flatten()
        ax_o, ay_o = own_accel.flatten()

        a_or = -ax_o * np.cos(beta) - ay_o * np.sin(beta)
        a_ob =  ax_o * np.sin(beta) - ay_o * np.cos(beta)

        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[1, 0] = -zeta * a_or
        A[1, 1] = -2.0 * rho
        A[1, 2] = -2.0 * beta_dot
        A[1, 3] = a_ob

        A[2, 0] = zeta * a_ob
        A[2, 1] = 2.0 * beta_dot
        A[2, 2] = -2.0 * rho
        A[2, 3] = a_or

        A[3, 2] = -zeta
        A[3, 3] = -rho

        return A

    def predict(self, dt: float, own_accel: np.ndarray = np.array([0.0, 0.0])):
        k1 = self._dynamics(self.y, own_accel)
        k2 = self._dynamics(self.y + 0.5 * dt * k1, own_accel)
        k3 = self._dynamics(self.y + 0.5 * dt * k2, own_accel)
        k4 = self._dynamics(self.y + dt * k3, own_accel)
        
        self.y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.y[0, 0] = self._wrap_angle(self.y[0, 0])

        A = self._system_jacobian(self.y, own_accel)
        F = np.eye(4) + A * dt + 0.5 * (A @ A) * (dt**2)

        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement_value: np.ndarray, own_vel: np.ndarray = None, measurement_type: str = 'B'):
        z = np.array(measurement_value, dtype=float).reshape(-1, 1)
        z_pred = self.h(self.y, own_vel=own_vel, measurement_type=measurement_type)

        y_res = z - z_pred
        y_res[0, 0] = self._wrap_angle(y_res[0, 0])
        if measurement_type == 'BRCS':
            y_res[2, 0] = self._wrap_angle(y_res[2, 0])

        H = self.H_jacobian(self.y, own_vel=own_vel, measurement_type=measurement_type)
        R = self.R_dict[measurement_type]

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.y = self.y + K @ y_res
        self.y[0, 0] = self._wrap_angle(self.y[0, 0])

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T


# =====================================================================
# 2. SIMULATION SETUP
# =====================================================================
dt = 1.0
total_time = 15 * 60  # 15 minutes (900 seconds)
steps = int(total_time / dt)

# Maneuver Timing
maneuver_start = 120.0
maneuver_end = 150.0
turn_duration = maneuver_end - maneuver_start

# Target Ground Truth (Constant Velocity)
target_pos = np.array([-3000.0, -4000.0])  # [East, North] in meters
target_vel = np.array([-5.0, 2.0])         # [vx, vy] in m/s

# Own-ship Setup
own_pos = np.array([0.0, 0.0])             # Starts at origin
own_speed = 8.0                             # 8 m/s (~15.5 knots)
heading_deg = 0.0                           # Initial heading East (0 deg)
turn_rate_deg = 90.0 / turn_duration        # 3 deg/s turn left toward North
turn_rate_rad = np.radians(turn_rate_deg)

# Measurement Noise Setup
sigma_B_rad = np.radians(0.2)
sigma_R = 5.0
sigma_C_rad = np.radians(1.0)
sigma_S = 0.2

R_dict = {
    'B': np.array([[sigma_B_rad**2]]),
    'BR': np.diag([sigma_B_rad**2, sigma_R**2]),
    'BRCS': np.diag([sigma_B_rad**2, sigma_R**2, sigma_C_rad**2, sigma_S**2])
}

# Initial MPC State Construction (Initial Guess)
# Correct initial bearing, but range guess is wrong (2000m vs ~5000m actual)
init_bearing = np.arctan2(target_pos[1], target_pos[0])
init_range_guess = 2000.0
init_target_pos_guess = np.array([
    init_range_guess * np.cos(init_bearing),
    init_range_guess * np.sin(init_bearing)
])
init_target_vel_guess = np.array([-4.0, 1.5])
init_own_vel = np.array([own_speed, 0.0])

y0_guess = MPCEKF.cartesian_to_mpc(
    target_pos=init_target_pos_guess,
    target_vel=init_target_vel_guess,
    own_pos=own_pos,
    own_vel=init_own_vel
)

# MPC Covariance and Process Noise
P0 = np.diag([
    np.radians(2.0)**2,  # beta variance (rad^2)
    (1e-3)**2,           # beta_dot variance ((rad/s)^2)
    (1e-2)**2,           # rho variance ((1/s)^2)
    (1.0 / 1000.0)**2    # zeta variance ((1/m)^2)
])

Q = np.diag([1e-7, 1e-8, 1e-8, 1e-10])  # Low process noise for pure CV target

ekf = MPCEKF(y0=y0_guess, P0=P0, Q=Q, R_dict=R_dict)

# Logging
history_target_true = []
history_own_ship = []
history_ekf_cartesian = []
history_pos_error = []
history_is_maneuvering = []

# =====================================================================
# 3. MAIN SIMULATION LOOP
# =====================================================================
np.random.seed(42)

for i in range(steps):
    t = i * dt
    is_maneuvering = maneuver_start <= t <= maneuver_end

    # 1. Update Own-ship Kinematics & Compute Acceleration
    if is_maneuvering:
        heading_deg += turn_rate_deg * dt
        heading_rad = np.radians(heading_deg)
        own_vel = np.array([own_speed * np.cos(heading_rad), own_speed * np.sin(heading_rad)])
        # Centripetal acceleration during turn: a = v * w * [-sin(theta), cos(theta)]
        own_accel = np.array([
            -own_speed * turn_rate_rad * np.sin(heading_rad),
             own_speed * turn_rate_rad * np.cos(heading_rad)
        ])
    else:
        heading_rad = np.radians(heading_deg)
        own_vel = np.array([own_speed * np.cos(heading_rad), own_speed * np.sin(heading_rad)])
        own_accel = np.array([0.0, 0.0])

    own_pos += own_vel * dt

    # 2. Update Target Ground Truth
    target_pos += target_vel * dt

    # 3. EKF Predict Step (Includes Own-ship Acceleration)
    ekf.predict(dt=dt, own_accel=own_accel)

    # 4. EKF Update Step (PAUSED DURING MANEUVER)
    if not is_maneuvering:
        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]
        true_bearing_rad = np.arctan2(dy, dx)
        noisy_bearing_rad = true_bearing_rad + np.random.normal(0, sigma_B_rad)

        ekf.update(measurement_value=[noisy_bearing_rad], own_vel=own_vel, measurement_type='B')

    # Convert state to Cartesian for logging
    target_est_cartesian = ekf.to_cartesian(own_pos=own_pos, own_vel=own_vel)

    history_target_true.append(target_pos.copy())
    history_own_ship.append(own_pos.copy())
    history_ekf_cartesian.append(target_est_cartesian.copy())
    history_is_maneuvering.append(is_maneuvering)

    # Position Error
    pos_err = np.linalg.norm(target_pos - target_est_cartesian[:2])
    history_pos_error.append(pos_err)

# =====================================================================
# 4. VISUALIZATION
# =====================================================================
target_true = np.array(history_target_true)
own_ship = np.array(history_own_ship)
ekf_est = np.array(history_ekf_cartesian)
time_vec = np.arange(steps) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: 2D Spatial Trajectory
ax1.plot(own_ship[:, 0], own_ship[:, 1], 'b-', label='Own-ship Path')
maneuver_idx = np.where(history_is_maneuvering)[0]
ax1.plot(own_ship[maneuver_idx, 0], own_ship[maneuver_idx, 1], 'r-', linewidth=3, label='Own-ship Maneuver (Updates Paused)')
ax1.plot(target_true[:, 0], target_true[:, 1], 'g--', label='Target True Path')
ax1.plot(ekf_est[:, 0], ekf_est[:, 1], 'm:', label='MPC EKF Estimated Path')
ax1.scatter(own_ship[0, 0], own_ship[0, 1], color='blue', marker='o', label='Own-ship Start')
ax1.scatter(target_true[0, 0], target_true[0, 1], color='green', marker='o', label='Target Start')
ax1.set_title('MPC EKF: 2D Target Tracking Trajectory')
ax1.set_xlabel('East [m]')
ax1.set_ylabel('North [m]')
ax1.grid(True)
ax1.legend()

# Plot 2: Position Estimation Error Over Time
ax2.plot(time_vec, history_pos_error, 'k-', label='Position Error (m)')
ax2.axvspan(maneuver_start, maneuver_end, color='red', alpha=0.2, label='Maneuver Window')
ax2.set_title('MPC EKF: Target Estimation Error vs Time')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position Error [m]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

    ======================================================
import numpy as np

def create_rotated_cartesian_p0(
    r0: float, 
    bearing_rad: float, 
    sigma_r: float, 
    sigma_bearing_rad: float, 
    sigma_v_radial: float = 2.0, 
    sigma_v_cross: float = 2.0
) -> np.ndarray:
    """
    Constructs a 4x4 Cartesian covariance matrix P0 aligned with the Line of Bearing.
    
    Parameters:
      r0: Initial range estimate [m]
      bearing_rad: Angle from target relative to own-ship [rad]
      sigma_r: Standard deviation of range guess [m]
      sigma_bearing_rad: Standard deviation of bearing measurement [rad]
      sigma_v_radial: Radial velocity uncertainty [m/s]
      sigma_v_cross: Cross-range velocity uncertainty [m/s]
    """
    # 1. Local cross-range position uncertainty
    sigma_cross_range = r0 * sigma_bearing_rad

    # 2. Local covariance matrices in LOS frame [Radial, Cross-Range]
    P_pos_local = np.diag([sigma_r**2, sigma_cross_range**2])
    P_vel_local = np.diag([sigma_v_radial**2, sigma_v_cross**2])

    # 3. Rotation Matrix from LOS frame to Cartesian (East, North)
    cos_b = np.cos(bearing_rad)
    sin_b = np.sin(bearing_rad)
    
    R = np.array([
        [cos_b, -sin_b],
        [sin_b,  cos_b]
    ])

    # 4. Rotate local covariances into global Cartesian frame
    P_pos_cart = R @ P_pos_local @ R.T
    P_vel_cart = R @ P_vel_local @ R.T

    # 5. Assemble full 4x4 State Covariance Matrix
    P0 = np.zeros((4, 4))
    P0[0:2, 0:2] = P_pos_cart
    P0[2:4, 2:4] = P_vel_cart

    return P0


# =====================================================================
# EXAMPLE USAGE (r0 = 21,000 m scenario)
# =====================================================================
own_pos = np.array([0.0, 0.0])
target_pos_guess = np.array([-12610.0, -16780.0])  # ~21 km at ~233 degrees

dx = target_pos_guess[0] - own_pos[0]
dy = target_pos_guess[1] - own_pos[1]

r0 = np.hypot(dx, dy)                         # 21,000 m
bearing_rad = np.arctan2(dy, dx)              # ~ -2.21 rad (-126.87 deg)

# Parameters: 50% range uncertainty (10.5 km), 0.2 deg bearing error
P0 = create_rotated_cartesian_p0(
    r0=r0,
    bearing_rad=bearing_rad,
    sigma_r=10500.0,
    sigma_bearing_rad=np.radians(0.2),
    sigma_v_radial=3.0,
    sigma_v_cross=3.0
)
