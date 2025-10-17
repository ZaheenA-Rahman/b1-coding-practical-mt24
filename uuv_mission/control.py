import numpy as np
from scipy.optimize import minimize

class PDController:
    def __init__(self, kp=0.15, kd=0.6):
        self.kp = kp
        self.kd = kd

    def get_action(self, t: int, positions: np.ndarray, mission, plant) -> float:
        error = mission.reference - positions.T[1]
        action = self.kp * error[t] + self.kd * (error[t] - (error[t-1] if t > 0 else 0))
        return action

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
    
