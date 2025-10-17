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

class MPCController:
    def __init__(self, horizon=5, Q=10, R=1e-4, n_hidden_layers=2, n_neurons=32):
        self.horizon = horizon  # Prediction horizon
        self.Q = Q  # State cost weight
        self.R = R  # Control cost weight
        self.PDController = PDController()  # Fallback controller
        self.transition_model = None  # Placeholder for neural network model
    
    def transition(self, position, velocity, action):
        pos_x = position[0] + velocity[0]
        pos_y = position[1] + velocity[1]

        force_y = -0.1 * velocity[1] + action
        acc_y = force_y
        vel_x = velocity[0]
        vel_y = velocity[1] + acc_y

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
        return float(result.x[0])
    
