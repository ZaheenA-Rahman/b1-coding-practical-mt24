from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

def plot_multiple_trajectories(trajectories, names, mission: Mission):

    assert len(trajectories) == len(names), "Number of trajectories and names must match."
    
    x_values = np.arange(len(mission.reference))
    min_depth = np.min(mission.cave_depth)
    max_height = np.max(mission.cave_height)

    plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
    plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                        color='saddlebrown', alpha=0.3)
    plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                        color='saddlebrown', alpha=0.3)
    for trajectory, name in zip(trajectories, names):
        plt.plot(trajectory.position[:, 0], trajectory.position[:, 1], label='Trajectory: ' + name)
    plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
    plt.legend(loc='lower center')
    plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """Create a Mission from a CSV file.

        The CSV is expected to have three columns: reference,
        cave_height, and cave_depth.

        Args:
            file_name: Path to CSV file.

        Returns:
            Mission: instance with three numpy.ndarray fields.
        """

        df = pd.read_csv(file_name)
        reference = df.iloc[:, 0].to_numpy()
        cave_height = df.iloc[:, 1].to_numpy()
        cave_depth = df.iloc[:, 2].to_numpy()

        return cls(reference, cave_height, cave_depth)

class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        velocities = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            velocities[t] = (self.plant.vel_x, self.plant.vel_y)
            try:
                actions[t] = self.controller.get_action(t, positions, velocities, actions, mission)
            except TypeError:
                actions[t] = self.controller.get_action(t, positions, mission, self.plant)
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
