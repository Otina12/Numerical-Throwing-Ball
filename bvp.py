from collections import deque
import numpy as np

k = 0.000001
rho = 1.22
area = 0.03
m = 2
dt = 0.01

class BVPSolver:
    def __init__(self, g, time, ball_radius):
        self.alpha = (k * rho * area) / (2 * m)
        self.time = time
        self.g = g # we change g to make the animation more realistic
        self.ball_radius = ball_radius

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

    def get_trajectory(self, init_pos, v0, t):
        x0, y0 = init_pos
        vx0, vy0 = v0
        y_cur = np.array([x0, vx0, y0, vy0])

        t_values = np.arange(0, t + dt, dt)
        trajectory = [y_cur]

        for _ in t_values[1:]:
            y_cur = self.rk4_step(y_cur, dt)
            trajectory.append(y_cur)

        trajectory = np.array(trajectory)
        return t_values, trajectory[:, 0], trajectory[:, 2] # times, x positions, y positions

    def get_velocities(self, init_pos, target_pos, t, tol = 1e-6, max_iter = 1000):
        xt, yt, _ = target_pos

        def newton_iteration(v):
            def F(x):
                x0, x1 = x
                _, x, y = self.get_trajectory(init_pos, (x0, x1), t)
                return np.array([x[-1] - xt, y[-1] - yt])
            
            error = F(v)
            h = 0.0001
            jacobian = [[0.01, 0.01], [0.01, 0.01]]

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
        
        vx_guess = dx / t
        vy_guess = (dy + 0.5 * self.g * t ** 2) / t

        v = np.array([vx_guess, vy_guess])

        for _ in range(max_iter):
            v_next = newton_iteration(v)
            if np.linalg.norm(v_next - v) < tol:
                break
            v = v_next
            
        # if it doesn't converge after 1000 iterations, it should be close enough
        return v[0], v[1]

    def shooting_method(self, targets, initial_point = (0, 0)):
        n = len(targets)
        x0, y0 = initial_point
        
        trajectories_ordered = []
        targets_ordered = []
        stack = deque([(target_i, self.time) for target_i in range(n-1, -1, -1)])
        targets_ordered = []
        visited = [False] * n
        memo = {}
        
        while stack:
            cur_target_i, time_to_hit = stack.pop()
            
            if visited[cur_target_i]:
                continue
            
            if cur_target_i in memo:
                trajectories_ordered.append(memo[cur_target_i])
                targets_ordered.append(cur_target_i)
                visited[cur_target_i] = True
                continue
            
            cur_target = targets[cur_target_i]
            vx, vy = self.get_velocities((x0, y0), cur_target, time_to_hit)
            t, x, y = self.get_trajectory((x0, y0), (vx, vy), time_to_hit)
            
            trajectory = list(zip(t, x, y))
            print(f"Target {cur_target}: vx0 = {vx:.3f} m/s, vy0 = {vy:.3f} m/s")
            
            intersecting_targets = self.find_intersecting_targets(trajectory, targets, cur_target_i)
            
            stack.append((cur_target_i, self.time)) # if intersecting_targets was empty, this will be added to trajectories in the next iteration
            for inters in intersecting_targets[::-1]:
                stack.append(inters)
           
            memo[cur_target_i] = trajectory
            
        return trajectories_ordered, [targets[i] for i in targets_ordered]
    
    def find_intersecting_targets(self, trajectory, targets, cur_target_i):
        intersections = []
        intersecting_targets = set()
        
        def intersect(x1, y1, r1, x2, y2, r2):
            distance_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
            radius_sq = (r1 + r2) ** 2
            return distance_sq <= radius_sq

        for i in range(0, len(trajectory), 5):
            t, x, y = trajectory[i]
            
            for tar_i, target in enumerate(targets):
                tar_x, tar_y, tar_r = target
                
                if tar_i != cur_target_i:
                    if tar_i not in intersecting_targets and intersect(x, y, self.ball_radius, tar_x, tar_y, tar_r):
                        intersections.append((tar_i, t + 0.2)) # 0.2 because first point it touches is not the center
                        intersecting_targets.add(tar_i)
                        
        return intersections