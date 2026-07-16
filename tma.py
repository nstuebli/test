import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# Extended Kalman Filter for bearing-only tracking (moving own-ship)
# State: target in global Cartesian coordinates: [x, y, vx, vy]^T
# Measurement: bearing from own-ship position (x_o, y_o)
# ============================================================

class BearingOnlyEKF:
    def __init__(self, dt, Q, R, P0, x0):
        self.dt = dt

        # State vector
        self.x = np.array(x0, dtype=float).reshape(4, 1)

        # Covariance
        self.P = np.array(P0, dtype=float)

        # Transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Noise matrices
        self.Q = np.array(Q, dtype=float)
        self.R = np.array([[R]], dtype=float)

    def h(self, x, own_pos):
        px, py = x[0, 0], x[1, 0]
        x_o, y_o = own_pos
        return math.atan2(py - y_o, px - x_o)
    
    def h_discrete(self, x, own_pos):
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos

        dx = px - x_o
        dy = py - y_o

        bearing = math.atan2(dy, dx)
        rng = math.sqrt(dx*dx + dy*dy)
        speed = math.sqrt(vx*vx + vy*vy)
        course = math.atan2(vy, vx)

        return np.array([[bearing], [rng], [speed], [course]])

    def H_jacobian(self, x, own_pos):
        px, py = x[0, 0], x[1, 0]
        x_o, y_o = own_pos
        dx = px - x_o
        dy = py - y_o
        r2 = dx*dx + dy*dy
        if r2 < 1e-8:
            r2 = 1e-8
        return np.array([[-dy/r2, dx/r2, 0, 0]])
        
    def H_discrete(self, x, own_pos):
        px, py, vx, vy = x.flatten()
        x_o, y_o = own_pos

        dx = px - x_o
        dy = py - y_o
        r2 = dx*dx + dy*dy
        r = math.sqrt(r2)
        v = math.sqrt(vx*vx + vy*vy)

        # Bearing
        Hb = np.array([-dy/r2, dx/r2, 0, 0])

        # Range
        Hr = np.array([dx/r, dy/r, 0, 0])

        # Speed
        Hs = np.array([0, 0, vx/v, vy/v])

        # Course
        Hc = np.array([0, 0, -vy/(v*v), vx/(v*v)])

        H = np.vstack([Hb, Hr, Hs, Hc])
        return H


    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bearing_meas, own_pos):
        z = np.array([[bearing_meas]])
        z_pred = np.array([[self.h(self.x, own_pos)]])
        y = z - z_pred
        y[0, 0] = (y[0, 0] + math.pi) % (2*math.pi) - math.pi

        H = self.H_jacobian(self.x, own_pos)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_discrete(self, z, own_pos, R_discrete):
        z = np.array(z).reshape(4,1)
        z_pred = self.h_discrete(self.x, own_pos)

        y = z - z_pred
        y[0,0] = (y[0,0] + math.pi) % (2*math.pi) - math.pi
        y[3,0] = (y[3,0] + math.pi) % (2*math.pi) - math.pi

        H = self.H_discrete(self.x, own_pos)
        S = H @ self.P @ H.T + R_discrete
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P



# ============================================================
# Simulation functions
# ============================================================

def run_simulation(dt,
                   steps,
                   target_init_pos,
                   target_vel,
                   ownship_motion_fn,
                   x0,
                   P0,
                   Q,
                   R,
                   meas_pause_before=0,
                   meas_pause_after=0):

    def target_motion(k):
        t = k * dt
        x = target_init_pos[0] + target_vel[0] * t
        y = target_init_pos[1] + target_vel[1] * t
        return x, y

    ekf = BearingOnlyEKF(dt=dt, x0=x0, P0=P0, Q=Q, R=R)

    true_track = []
    est_track = []
    own_track = []

    last_maneuver_time = -1e9  # effectively "never"

    # discrete high-accuracy measurements
    discrete_times = [500, 1200, 2000, 3000]
    R_discrete = np.diag([
        (0.05 * math.pi/180)**2,  # bearing variance
        5.0**2,                   # range variance
        0.2**2,                   # speed variance
        (1.0 * math.pi/180)**2    # course variance
    ])

    for k in range(steps):
        x_t, y_t = target_motion(k)
        x_o, y_o, vx_o, vy_o, is_maneuvering = ownship_motion_fn(k, dt)

        t = k * dt

        if is_maneuvering:
            last_maneuver_time = t

        blackout = (
            (t >= last_maneuver_time - meas_pause_before) and
            (t <= last_maneuver_time + meas_pause_after)
        )

        dx = x_t - x_o
        dy = y_t - y_o
        bearing_true = math.atan2(dy, dx)

        if blackout:
            bearing_meas = None
        else:
            bearing_meas = bearing_true + np.random.normal(0, math.sqrt(R))

        ekf.predict()
        if bearing_meas is not None:
            ekf.update(bearing_meas, (x_o, y_o))

        if k in discrete_times:
            z = [
                bearing_true,
                math.sqrt(dx*dx + dy*dy),
                math.sqrt(target_vel[0]**2 + target_vel[1]**2),
                math.atan2(target_vel[1], target_vel[0])
            ]
            ekf.update_discrete(z, (x_o, y_o), R_discrete)

        true_track.append((x_t, y_t))
        est_track.append((ekf.x[0, 0], ekf.x[1, 0]))
        own_track.append((x_o, y_o))

    true_track = np.array(true_track)
    est_track = np.array(est_track)
    own_track = np.array(own_track)

    plt.figure(figsize=(10, 8))
    plt.plot(true_track[:, 0], true_track[:, 1], 'b-', label="True Target")
    plt.plot(est_track[:, 0], est_track[:, 1], 'r--', label="Estimated Target")
    plt.plot(own_track[:, 0], own_track[:, 1], 'k-', label="Ownship")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def improved_zigzag_ownship_motion(k, dt):
        speed = 4.0
        intervals = [400, 700, 500, 200]
        headings_deg = [0, 35, -25, 90]
        headings = [math.radians(h) for h in headings_deg]

        t = k * dt
        cumulative = np.cumsum(intervals)
        segment = np.searchsorted(cumulative, t)

        heading = headings[segment % len(headings)]
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

        x = 0.0
        y = 0.0
        for i in range(segment):
            h = headings[i % len(headings)]
            x += speed * math.cos(h) * intervals[i]
            y += speed * math.sin(h) * intervals[i]

        if segment < len(intervals):
            t_rem = t - (cumulative[segment - 1] if segment > 0 else 0)
        else:
            cycle_time = cumulative[-1]
            t_cycle = t % cycle_time
            segment = np.searchsorted(cumulative, t_cycle)
            heading = headings[segment]
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

            x = 0.0
            y = 0.0
            for i in range(segment):
                h = headings[i]
                x += speed * math.cos(h) * intervals[i]
                y += speed * math.sin(h) * intervals[i]

            t_rem = t_cycle - (cumulative[segment - 1] if segment > 0 else 0)

        x += vx * t_rem
        y += vy * t_rem

        is_maneuvering = any(abs(t - tc) < dt/2 for tc in cumulative)

        return x, y, vx, vy, is_maneuvering


if __name__ == "__main__":
    # --- CONSTANTS ---
    DT = 1.0
    STEPS = 600 * 4

    TARGET_INIT_POS = (0.0, 3000.0)
    TARGET_VEL = (8.0, 1.0)

    # EKF initial guess
    INIT_RANGE = 5000.0
    INIT_BEARING = math.atan2(
        TARGET_INIT_POS[1],
        TARGET_INIT_POS[0]
    )

    X0 = [
        INIT_RANGE * math.cos(INIT_BEARING),
        INIT_RANGE * math.sin(INIT_BEARING),
        7.0,   # vx guess
        2.0    # vy guess
    ]

    # EKF covariance
    P0 = np.diag([5e3, 5e3, 1.0, 1.0])

    # EKF noise
    Q = np.diag([10.0, 10.0, 0.0, 0.0])
    R = (0.2 * math.pi / 180.0)**2

    run_simulation(
        dt=DT,
        steps=STEPS,
        target_init_pos=TARGET_INIT_POS,
        target_vel=TARGET_VEL,
        ownship_motion_fn=improved_zigzag_ownship_motion,
        x0=X0,
        P0=P0,
        Q=Q,
        R=R,
        meas_pause_before=5.0,
        meas_pause_after=15.0
    )
