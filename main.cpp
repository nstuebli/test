class BearingOnlyEKF:
    """
        Creates EKF object with target state x [x, y, vx, vy] in cartesian coordinates
        and measurements in bearing relative to own-ship position.
        Args:
            dt (float): First number
            Q (np.array[float]): Process noise covariance
            R (np.array[float]): Observation noise covariance
            P0 (np.array[float]): Initial estimation covariance
            x0 (np.array[float]): Initial estimation mean
    """
    def __init__(self, dt, x0, P0, Q, R):
        self.dt = dt

        # State vector of estimation, init with x0
        self.x = np.array(x0, dtype=float).reshape(4, 1)

        # Covariance of estimation, init with P0
        self.P = np.array(P0, dtype=float)

        # Transition matrix for constant velocity modell
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Process Q and measrument noise R
        self.Q = np.array(Q, dtype=float)

        self.R_B = R[0]
        self.R_BR = np.vstack([self.R_B, R[1]])
        self.R_BRCS = np.vstack([self.R_B, R[1], R[2], R[3]])

    def h(self, x, own_pos, measurement_type='B'):
        """
        Observation model. Maps state vector to observation space.
        Args:
            x (np.array[float]): Target state vector [px, py, vx, vy].
            own_pos (np.array[float]): Ownship position [x_o, y_o].
            measurement_type (str): 'B' (Bearing), 'BR' (Bearing-Range), 'BRCS' (Bearing-Range-Course-Speed).
        Returns:
            np.array: Predicted measurement vector.
        """
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos

        dx = px - x_o
        dy = py - y_o

        if measurement_type == 'B':
            bearing = math.atan2(dy, dx)
            return np.array([bearing])
            
        elif measurement_type == 'BR':
            bearing = math.atan2(dy, dx)
            rng = math.sqrt(dx*dx + dy*dy)
            return np.array([bearing, rng])
            
        elif measurement_type == 'BRCS':
            bearing = math.atan2(dy, dx)
            rng = math.sqrt(dx*dx + dy*dy)
            speed = math.sqrt(vx*vx + vy*vy)
            course = math.atan2(vy, vx)
            return np.array([bearing, rng, speed, course])

    def H_jacobian(self, x, own_pos, measurement_type='B'):
        """
            Jacobian for observation model h() for bearing only measurement
            Args:
                x (np.array[float]): State vector [m, m, m/s, m/s]
                own_pos (np.array[float]): Cartesian position of own ship [m, m]
            Returns:
                np.array[float]: Jacobian matrix
        """
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos
        dx = px - x_o
        dy = py - y_o

        if (measurement_type == 'B'):
            r2 = dx*dx + dy*dy
            if r2 < 1e-8:
                r2 = 1e-8
            Hb = np.array([-dy/r2, dx/r2, 0, 0])
            return Hb
        elif (measurement_type == 'BR'):
            r2 = dx*dx + dy*dy
            if r2 < 1e-8:
                r2 = 1e-8
            rng = math.sqrt(r2)
            Hb = np.array([-dy/r2, dx/r2, 0, 0])
            Hr = np.array([dx/r, dy/r, 0, 0])
            return np.vstack([Hb, Hr])
        elif (measurement_type == 'BRCS'):
            r2 = dx*dx + dy*dy
            if r2 < 1e-8:
                r2 = 1e-8
            rng = math.sqrt(r2)
            v = math.sqrt(vx*vx + vy*vy)
            Hb = np.array([-dy/r2, dx/r2, 0, 0])
            Hr = np.array([dx/r, dy/r, 0, 0])
            Hc = np.array([0, 0, -vy/(v*v), vx/(v*v)])
            Hs = np.array([0, 0, vx/v, vy/v])
            return np.vstack([Hb, Hr, Hc, Hs])

    def predict(self):
        """
            Propagation of state for one timestep
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_value, own_pos, measurement_type='B'):
        """
            Apply bearing only measurement to state
            Args:
                bearing_meas (float): XXX
                own_pos (np.array[float]): XXX
        """
        z = measurement_value
        z_predicted = self.h(self.x, own_pos, measurement_type)
        y = z - z_predicted
        H = self.H_jacobian(self.x, own_pos, measurement_type)

        if (measurement_type == 'B'):
            y = (y + math.pi) % (2*math.pi) - math.pi
            S = H @ self.P @ H.T + self.R_B
        elif (measurement_type == 'BR'):
            y[0] = (y[0] + math.pi) % (2*math.pi) - math.pi
            S = H @ self.P @ H.T + self.R_BR
        elif (measurement_type == 'BRCS'):
            y[0] = (y[0] + math.pi) % (2*math.pi) - math.pi
            y[3] = (y[3] + math.pi) % (2*math.pi) - math.pi
            S = H @ self.P @ H.T + self.R_BRCS
        
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
