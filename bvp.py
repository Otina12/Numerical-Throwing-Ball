import numpy as np

k = 0.000001
rho = 1.22
area = 0.03
m = 2
dt = 0.01

class BVPSolver:
    def __init__(self, g, time):
        self.alpha = (k * rho * area) / (2 * m)
        self.time = time
        self.g = g # we change g to make the animation more realistic

    def ode_system(self, y):
        x_pos, vx, y_pos, vy = y

        speed = np.sqrt(vx ** 2 + vy ** 2)
        drag_factor = self.alpha * speed

        dxdt = vx
        dvxdt = -drag_factor * vx
        dydt = vy
        dvydt = -self.g - drag_factor * vy

        return np.array([dxdt, dvxdt, dydt, dvydt])

    def euler_step(self, y, dt):
        dy = self.ode_system(y)
        y_next = y + dt * dy
        return y_next
        
    def rk4_step(self, y, dt):
        k1 = self.ode_system(y)
        k2 = self.ode_system(y + 0.5 * dt * k1)
        k3 = self.ode_system(y + 0.5 * dt * k2)
        k4 = self.ode_system(y + dt * k3)

        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_trajectory(self, init_pos, v0):
        x0, y0 = init_pos
        vx0, vy0 = v0
        y_cur = np.array([x0, vx0, y0, vy0])

        t_values = np.arange(0, self.time + dt, dt)
        trajectory = [y_cur]

        for _ in t_values[1:]:
            y_cur = self.rk4_step(y_cur, dt)
            trajectory.append(y_cur)

        trajectory = np.array(trajectory)
        return t_values, trajectory[:, 0], trajectory[:, 2] # times, x positions, y positions

    def get_velocities(self, init_pos, target_pos, tol = 1e-6, max_iter = 1000):
        xt, yt, _ = target_pos

        def newton_iteration(v):
            def F(x):
                x0, x1 = x
                _, x, y = self.get_trajectory(init_pos, (x0, x1))
                return np.array([x[-1] - xt, y[-1] - yt])
            
            error = F(v)
            h = 0.0001
            jacobian = [[0, 0], [0, 0]]

            # we create the Jacobian Matrix by approximating partial derivatives for both x and y
            for i in range(2):
                v_changed = v.copy()
                v_changed[i] += h

                error_new = F(v_changed)
                
                for j in range(2):
                    jacobian[j][i] = (error_new[j] - error[j]) / h

            # x_next = x_cur - J^(-1)(x_cur)·F(x_cur)
            return v - np.linalg.inv(jacobian).dot(error)

        x0, y0 = init_pos
        dx, dy = xt - x0, yt - y0
        
        vx_guess = dx / self.time
        vy_guess = (dy + 0.5 * self.g * self.time ** 2) / self.time

        v = np.array([vx_guess, vy_guess])

        for _ in range(max_iter):
            v_next = newton_iteration(v)
            if np.linalg.norm(v_next - v) < tol:
                break
            v = v_next
            
        # if it doesn't converge after 1000 iterations, it should be close enough
        return v[0], v[1]

    def shooting_method(self, targets, initial_point=(0, 0)):
        x0, y0 = initial_point
        trajectories = []

        for target in targets:
            vx, vy = self.get_velocities((x0, y0), target)
            t, x, y = self.get_trajectory((x0, y0), (vx, vy))

            trajectory = list(zip(t, x, y))
            trajectories.append(trajectory)
            print(f"Target {target}: vx0 = {vx:.3f} m/s, vy0 = {vy:.3f} m/s")

        return trajectories