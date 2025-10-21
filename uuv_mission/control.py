import numpy as np
from scipy.optimize import minimize

class PDController:
    def __init__(self, kp=0.15, kd=0.6):
        self.kp = kp
        self.kd = kd

    def get_action(self, t: int, positions: np.ndarray, velocities: np.ndarray, actions: np.ndarray, mission) -> float:
        error = mission.reference - positions.T[1]
        action = self.kp * error[t] + self.kd * (error[t] - (error[t-1] if t > 0 else 0))
        return action

class PIDController:
    """PID controller with integral anti-windup.

    get_action signature matches PDController.get_action:
        get_action(t, positions, velocities, actions, mission) -> float

    Notes:
    - The controller integrates the vertical error (reference - depth).
    - Anti-windup is implemented by clamping the integral term to a configurable range.
    - A reset() method clears the integral state.
    """
    def __init__(self, kp: float = 0.15, ki: float = 0.01, kd: float = 0.6, integral_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)

        # internal state
        self._integral = 0.0

    def reset(self) -> None:
        """Reset the integral term (useful between missions)."""
        self._integral = 0.0

    def get_action(self, t: int, positions: np.ndarray, velocities: np.ndarray, actions: np.ndarray, mission) -> float:
        # Compute error time-series (reference - current depth)
        ref = mission.reference
        depths = positions.T[1]

        # Current error and derivative
        err = ref[t] - depths[t]
        d_err = 0.0 if t == 0 else (err - (ref[t-1] - depths[t-1]))

        # Integrate with trapezoidal update (approx)
        if t == 0:
            self._integral = err
        else:
            prev_err = ref[t-1] - depths[t-1]
            self._integral += 0.5 * (err + prev_err)

        # Anti-windup via clamping
        self._integral = max(-self.integral_limit, min(self._integral, self.integral_limit))

        # PID output
        action = self.kp * err + self.ki * self._integral + self.kd * d_err

        return float(action)

class MPCController:
    def __init__(self, horizon=5, Q=10, R=1e-4):
        self.horizon = horizon  # Prediction horizon
        self.Q = Q  # State cost weight
        self.R = R  # Control cost weight
        
    def predict_trajectory(self, initial_state, actions, plant):
        # Simulate the system forward over the horizon
        trajectory = np.zeros((self.horizon + 1, 2))
        trajectory[0] = initial_state
        
        # Store original plant state
        orig_x, orig_y = plant.pos_x, plant.pos_y
        orig_vx, orig_vy = plant.vel_x, plant.vel_y
        
        # Simulate forward
        for t in range(self.horizon):
            plant.transition(actions[t], 0)  # Assume no disturbance in prediction
            trajectory[t + 1] = plant.get_position()
            
        # Restore plant state
        plant.pos_x, plant.pos_y = orig_x, orig_y
        plant.vel_x, plant.vel_y = orig_vx, orig_vy
        
        return trajectory

    def objective(self, actions, t, positions, mission, plant):
        # Get current state
        current_state = positions[t]
        
        # Predict future trajectory
        trajectory = self.predict_trajectory(current_state, actions, plant)
        
        # Calculate cost over prediction horizon
        cost = 0
        for i in range(self.horizon):
            # Reference tracking error cost
            if t + i < len(mission.reference):
                ref_error = mission.reference[t + i] - trajectory[i, 1]
                cost += self.Q * ref_error**2
            
            # Control effort cost
            cost += self.R * actions[i]**2
            
            # Add barrier terms for cave constraints
            if t + i < len(mission.reference):
                height_violation = trajectory[i, 1] - mission.cave_height[t + i]
                depth_violation = mission.cave_depth[t + i] - trajectory[i, 1]
                if height_violation > 0:
                    cost += 1000 * height_violation**2
                if depth_violation > 0:
                    cost += 1000 * depth_violation**2
        
        return cost

    def get_action(self, t: int, positions: np.ndarray, mission, plant) -> float:
        # Initial guess: zero control sequence
        initial_guess = np.zeros(self.horizon)
        
        # Optimize control sequence
        result = minimize(
            self.objective,
            initial_guess,
            args=(t, positions, mission, plant),
            method='SLSQP',
            bounds=[(-10, 10)] * self.horizon  # Limit control magnitude
        )
        
        # Return only the first control action
        return float(result.x[0])

class UMPCController:
    def __init__(self, horizon=5, Q=10, R=1e-4, kp=0.15, kd=0.6):
        self.horizon = horizon  # Prediction horizon
        self.Q = Q  # State cost weight
        self.R = R  # Control cost weight
        self.FallbackController = PDController(kp, kd)  # Fallback controller
        self.transition_model = None  # Placeholder for SLR transition model
        self.transition_model_B = None

    def update_transition_model(self, positions, velocities, actions, t):
        posti = positions[:t]
        velti = velocities[:t]
        actt = actions[:t]

        postf = positions[1:t+1]
        veltf = velocities[1:t+1]

        X = np.array([posti[:, 0], velti[:, 0], actt]).T
        Y = np.array([postf[:, 0], veltf[:, 0]]).T
        Bx = np.linalg.pinv(X.T @ X) @ X.T @ Y

        X = np.array([posti[:, 1], velti[:, 1], actt]).T
        Y = np.array([postf[:, 1], veltf[:, 1]]).T
        By = np.linalg.pinv(X.T @ X) @ X.T @ Y

        self.transition_model = lambda pos, vel, act: (*(np.array([pos[0], vel[0], act]) @ Bx), *(np.array([pos[1], vel[1], act]) @ By))

    def evaluate_transition_model(self, positions, velocities, actions, t):
        metric = 0
        for t_ in range(t):
            pos_pred_x, vel_pred_x, pos_pred_y, vel_pred_y = self.transition_model(positions[t_], velocities[t_], actions[t_])
            pos_x, pos_y, vel_x, vel_y = *positions[t_+1], *velocities[t_+1]
            err_norm = np.sqrt((pos_pred_x - pos_x)**2 + (pos_pred_y - pos_y)**2 + (vel_pred_x - vel_x)**2 + (vel_pred_y - vel_y)**2)
            val_norm = np.sqrt(pos_x**2 + pos_y**2 + vel_x**2 + vel_y**2)
            metric += err_norm / (val_norm + 1e-6)
        return metric / t

    def transition(self, position, velocity, action):
        pos_x, vel_x, pos_y, vel_y = self.transition_model(position, velocity, action)

        return (pos_x, pos_y), (vel_x, vel_y)

    def predict_trajectory(self, positions, velocities, actions, t):
        # Simulate the system forward over the horizon
        trajectory = np.zeros((self.horizon + 1, 2))
        vtrajectory = np.zeros((self.horizon + 1, 2))
        trajectory[0] = positions[t]
        vtrajectory[0] = velocities[t]
        
        for t_ in range(self.horizon):
            trajectory[t_ + 1], vtrajectory[t_ + 1] = self.transition(trajectory[t_], vtrajectory[t_], actions[t_])
        
        return trajectory

    def objective(self, actions, mission, positions, velocities, t):

        # Predict future trajectory
        trajectory = self.predict_trajectory(positions, velocities, actions, t)
        
        # Calculate cost over prediction horizon
        cost = 0
        for i in range(self.horizon):
            # Reference tracking error cost
            if t + i < len(mission.reference):
                ref_error = mission.reference[t + i] - trajectory[i, 1]
                cost += self.Q * ref_error**2
            
            # Control effort cost
            cost += self.R * actions[i]**2
            
            # Add barrier terms for cave constraints
            if t + i < len(mission.reference):
                height_violation = trajectory[i, 1] - mission.cave_height[t + i]
                depth_violation = mission.cave_depth[t + i] - trajectory[i, 1]
                if height_violation > 0:
                    cost += 1000 * height_violation**2
                if depth_violation > 0:
                    cost += 1000 * depth_violation**2
        
        return cost

    def get_action(self, t: int, positions: np.ndarray, velocities: np.ndarray, actions: np.ndarray, mission) -> float:

        if t < self.horizon:
            return self.FallbackController.get_action(t, positions, velocities, actions, mission)

        self.update_transition_model(positions, velocities, actions, t)
        metric = self.evaluate_transition_model(positions, velocities, actions, t)

        # Initial guess: zero control sequence
        initial_guess = np.zeros(self.horizon)
        
        # Optimize control sequence
        result = minimize(
            self.objective,
            initial_guess,
            args=(mission, positions, velocities, t),
            method='SLSQP',
            bounds=[(-10, 10)] * self.horizon  # Limit control magnitude
        )
        
        # Return only the first control action
        return float(result.x[0]) if metric < 0.2 else self.FallbackController.get_action(t, positions, velocities, actions, mission)
