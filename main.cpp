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
