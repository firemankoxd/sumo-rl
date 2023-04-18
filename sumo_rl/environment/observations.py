"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )


class DrqNorm(ObservationFunction):
    """todo"""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        self.measures_per_lane = dict()

    def __call__(self) -> np.ndarray:
        """todo"""
        ts_id = self.ts.id
        ts_phase = self.ts.sumo.trafficlight.getPhase(ts_id)
        vehicles = self.ts._get_veh_list()
        waiting_times = dict()
        obs = []

        for idx, lane in enumerate(self.ts.lanes):
            lane_obs = []

            # Active phase
            if idx == ts_phase:
                lane_obs.append(1)
            else:
                lane_obs.append(0)
            
            vehicles = []
            self.measures_per_lane[lane] = {'queue': 0, 'approach': 0, 'total_wait': 0, 'max_wait': 0}
            lane_vehicles = self.get_vehicles(lane, max_distance=200)

            for vehicle in lane_vehicles:
                if vehicle in waiting_times:
                    waiting_times[vehicle] += 10  #step_length
                elif self.ts.sumo.vehicle.getWaitingTime(vehicle) > 0:  # Vehicle stopped here, add it
                    waiting_times[vehicle] = self.ts.sumo.vehicle.getWaitingTime(vehicle)

                vehicle_measures = dict()
                vehicle_measures['id'] = vehicle
                vehicle_measures['wait'] = waiting_times[vehicle] if vehicle in waiting_times else 0
                vehicle_measures['speed'] = self.ts.sumo.vehicle.getSpeed(vehicle)
                vehicle_measures['acceleration'] = self.ts.sumo.vehicle.getAcceleration(vehicle)
                vehicle_measures['position'] = self.ts.sumo.vehicle.getLanePosition(vehicle)
                vehicle_measures['type'] = self.ts.sumo.vehicle.getTypeID(vehicle)
                vehicles.append(vehicle_measures)
                
                if vehicle_measures['wait'] > 0:
                    self.measures_per_lane[lane]['total_wait'] = self.measures_per_lane[lane]['total_wait'] + vehicle_measures['wait']
                    self.measures_per_lane[lane]['queue'] = self.measures_per_lane[lane]['queue'] + 1
                    if vehicle_measures['wait'] > self.measures_per_lane[lane]['max_wait']:
                        self.measures_per_lane[lane]['max_wait'] = vehicle_measures['wait']
                else:
                    self.measures_per_lane[lane]['approach'] = self.measures_per_lane[lane]['approach'] + 1

            # Approach
            lane_obs.append(self.measures_per_lane[lane]['approach'] / 28)
            
            #  Total wait
            lane_obs.append(self.measures_per_lane[lane]['total_wait'] / 28)
            
            # Queue
            lane_obs.append(self.measures_per_lane[lane]['queue'] / 28)

            # Speed
            total_speed = 0
            for idx, vehicle in enumerate(vehicles):
                total_speed += (vehicles[idx]['speed'] / 20 / 28)
            lane_obs.append(total_speed)

            obs.append(lane_obs)

        observation = np.concatenate(obs, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """todo"""
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.ts.lanes)*5,)
        )
        return spaces.Box(low=0.0, high=1.0, shape=(len(self.ts.lanes),5)) # TypeError: only size-1 arrays can be converted to Python scalars
    
    def get_vehicles(self, lane, max_distance):
        detectable = []
        for vehicle in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.ts.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable