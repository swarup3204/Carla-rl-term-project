import sys
import glob
try:
    '''sys.path.append(glob.glob('/home/user1/Downloads/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py2.7-linux-x86_64.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])'''
    sys.path.append(glob.glob(
        '/home/user1/Downloads/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla
import gymnasium as gym
from gymnasium import spaces
from time import sleep, time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


import queue


PORT = 3000


class PathFollower(gym.Env):
    metadata = {
        'reward_offset': 6,
        'truncated_penalty': 40,
        'terminated_penalty': 50,
        'coeff_velocity_error': 5,
        'coeff_deviation_error': 5,
        'coeff_heading_error': 10
    }

    def __init__(self, render_mode='human',
                 set_velocity=6,
                 path_length=50,
                 max_deviation=2):
        super().__init__()
        # print("initialised")
        self._set_velocity = set_velocity
        self._path_length = path_length
        self._max_deviation = max_deviation
        self._render_mode = render_mode

        # x, y, yaw, velocity, heading_error, deviation_error
        self.observation_space = spaces.Box(
            low=np.array([-250, -250, -180, -25, -np.pi, -10],
                         dtype=np.float32),
            high=np.array([250, 250, 180, 25, np.pi, 10], dtype=np.float32)
        )

        self.action_space = spaces.Box(
            low=np.array([0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32)
        )

        client = carla.Client('localhost', PORT)
        client.set_timeout(5.0)
        self._world = client.get_world()
        self._map = self._world.get_map()
        self._view = self._world.get_spectator()
        self._blueprint_lib = self._world.get_blueprint_library()
        self._map_spawn_points = self._map.get_spawn_points()

        self._index = 1
        self._errors = {
            'velocity_error': 0,
            'deviation_error': 0,
            'heading_error': 0
        }
        self.collided = False
        self.obs, self.info = None, {}
        self.path = {'x': [], 'y': [], 'yaw': []}
        self.trajectory = deepcopy(self.path)
        self.reward_history = [0]
        # self.prev_steer = 0

    def step(self, action):
        # Apply control to the vehicle using the action
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(action[0]),
            steer=float(action[1]),
            brake=float(action[2]),
        ))
        sleep(1.5)
        # self.prev_steer = float(action[1])
        # Update car's observation after taking action
        self._update_car_obs()
        print(f"Vehicle speed = {self.obs[3]}")

        # Optionally render the scene if required
        if self._render_mode == 'human':
            self._cam_follow_car()

        # Calculate the error metrics (velocity error, deviation error, heading error)
        self._calculate_error_metrics()

        # Calculate total error based on the different error metrics
        total_error = np.linalg.norm([
            self._errors['velocity_error'] *
            self.metadata['coeff_velocity_error'],
            self._errors['deviation_error'] *
            self.metadata['coeff_deviation_error'],
            self._errors['heading_error'] *
            self.metadata['coeff_heading_error']
        ])

        # Compute the reward based on the total error
        # reward = 1 / (total_error + 1e-2) - self.metadata['reward_offset']
        reward = 0
        # earlier it was not abs
        steer = float(action[1])
        if abs(steer) > 0:
            reward -= abs(steer)*20

        if total_error <= 20:
            reward += 25
        elif total_error <= 25:
            reward += 10 + (25 - total_error)
        elif total_error <= 50:
            reward -= 5
        else:
            reward -= 10

        # Initialize truncation and termination flags
        truncated = terminated = False

        # Check if the error is beyond the threshold for truncation or termination
        if self._errors['deviation_error'] > self._max_deviation or self._errors['heading_error'] > np.radians(90):
            truncated = True
            reward -= self.metadata['truncated_penalty']

        if self.collided or (len(self.reward_history) > 100 and self.obs[3] < 0.1):
            terminated = True
            reward -= self.metadata['terminated_penalty']

        # Calculate path coverage
        # path_coverage = self._calculate_path_coverage()
        # print(f"Path Coverage: {path_coverage:.2f}%")

        # Append the reward to the reward history
        self.reward_history.append(reward)
        self.collided = False  # Reset collision flag

        # Return the updated observation, reward, truncation and termination flags, and additional info
        plt.pause(0.01)  # Pause to allow visualization updates

        print("\n--- Step Info ---")
        print(f"Vehicle Speed: {self.obs[3]:.2f} m/s")
        print(f"Velocity Error: {self._errors['velocity_error']:.2f}, Deviation Error: {
              self._errors['deviation_error']:.2f}, Heading Error: {self._errors['heading_error']:.2f}, Total Error: {total_error:.2f}")
        print(f"Action Taken: Throttle = {action[0]:.2f}, Steer = {steer:.2f}, Brake = {action[2]:.2f}")
        print(f"Reward: {reward:.2f}")
        print("--- End of Step ---\n")

        return self.obs, reward, truncated, terminated, self.info

    def reset(self, seed=None):
        # Path coverage
        if len(self.path['x']) != 0:
            print("---- A training episode ended -----")
            path_coverage = (self._index / len(self.path['x'])) * 100
            print(f"### Path Coverage: {path_coverage:.2f}%")
        super().reset(seed=seed)
        self.close()
        self._index = 1
        self.collided = False
        self._spawn_ego_vehicle()
        self._update_car_obs()
        # plt.show()
        self._cam_follow_car()
        self._create_global_path()
        self._calculate_error_metrics()
        self.reward_history = [0]
        return self.obs, self.info

    def render(self, i):
        try:
            plt.gcf().clear()
            plt.subplot(211)
            plt.plot(self.path['x'], self.path['y'], '-y', lw=5, alpha=0.5)
            plt.plot(self.trajectory['x'], self.trajectory['y'], '--k')
            plt.plot(self.trajectory['x'][-1], self.trajectory['y'][-1],
                     'o' + 'cr'[self.collided], mec='k', ms=7)
            plt.subplot(212)
            plt.plot(self.reward_history, '-' +
                     'rg'[int(self.reward_history[-1] > 0)])
        except:
            print("")

    def _calculate_path_coverage(self):
        """Calculate the percentage of the path covered by the vehicle"""
        if not self.path['x']:
            return 0  # No path, return 0 coverage
        total_path_length = np.sum(
            np.sqrt(np.diff(self.path['x'])**2 + np.diff(self.path['y'])**2)
        )
        vehicle_travelled = np.sum(
            np.sqrt((np.array(self.trajectory['x']) - np.array(self.path['x'][0]))**2 +
                    (np.array(self.trajectory['y']) - np.array(self.path['y'][0]))**2)
        )
        return (vehicle_travelled / total_path_length) * 100

    def close(self):
        print("Close called")
        super().close()
        if self.obs is not None:
            self.camera.stop()
            self.camera.destroy()
            self.vehicle.destroy()
            self.collision_sensor.destroy()
        # plt.show()

    # ------------------------------ UTIL FUNCTIONS----------------------------------

    def _spawn_ego_vehicle(self):
        if self.obs is not None:
            self.vehicle.destroy()
            self.camera.destroy()
            self.collision_sensor.destroy()
        suc = 0
        while suc == 0:
            try:
                self.vehicle = self._world.spawn_actor(
                    self._blueprint_lib.find('vehicle.tesla.model3'),
                    np.random.choice(self._map_spawn_points)
                )
                suc = 1
            except:
                print("Failed spawn")
                pass
        # self.actor_list.append(self.vehicle)
        sleep(1)
        self.collision_sensor = self._world.spawn_actor(
            self._blueprint_lib.find(
                'sensor.other.collision'), carla.Transform(),
            attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.collided = False

        def collision_event():
            self.collided = True
        self.collision_sensor.listen(lambda flag: collision_event())

        camera_bp = self._blueprint_lib.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self._world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle)
        # self.actor_list.append(self.camera)
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)

    def _update_car_obs(self):
        veh_tf = self.vehicle.get_transform()
        vel_3D = self.vehicle.get_velocity()
        velocity = np.sqrt(vel_3D.x ** 2 + vel_3D.y ** 2)
        self.obs = np.array([
            veh_tf.location.x,
            veh_tf.location.y,
            veh_tf.rotation.yaw,
            velocity,
            self._errors['deviation_error'],
            self._errors['heading_error']],
            dtype=np.float32)
        self.trajectory['x'].append(self.obs[0])
        self.trajectory['y'].append(self.obs[1])
        self.trajectory['yaw'].append(self.obs[2])

    def _cam_follow_car(self):
        veh_tf = self.vehicle.get_transform()
        loc, rot = veh_tf.location, veh_tf.rotation
        dx = 10 * np.cos(np.radians(rot.yaw))
        dy = 10 * np.sin(np.radians(rot.yaw))
        view_tf = carla.Transform(
            carla.Location(
                x=loc.x-dx,
                y=loc.y-dy,
                z=loc.z+5),
            carla.Rotation(
                roll=rot.roll,
                pitch=rot.pitch-20,
                yaw=rot.yaw)
        )
        self._view.set_transform(view_tf)

    # def _create_global_path(self):
    #     self.path = {'x': [], 'y': [], 'yaw': []}
    #     self.trajectory = deepcopy(self.path)
    #     veh_waypoint = self._map.get_waypoint(
    #         location=self.vehicle.get_location(),
    #         project_to_road=True,
    #         lane_type=carla.LaneType.Driving
    #     )
    #     for i in range(self._path_length):
    #         veh_tf = veh_waypoint.transform
    #         self.path['x'].append(veh_tf.location.x)
    #         self.path['y'].append(veh_tf.location.y)
    #         self.path['yaw'].append(veh_tf.rotation.yaw)
    #         veh_waypoint = veh_waypoint.next(5.0)[0]

    def _create_global_path(self):
        """Creates a straight road path by following waypoints that have minimal turn deviation."""
        self.path = {'x': [], 'y': [], 'yaw': []}
        self.trajectory = deepcopy(self.path)

        # Get the initial waypoint based on the vehicle's starting location
        start_waypoint = self._map.get_waypoint(
            location=self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # Store the initial yaw to use as a reference for "straight" waypoints
        initial_yaw = start_waypoint.transform.rotation.yaw

        current_waypoint = start_waypoint
        for _ in range(self._path_length):
            transform = current_waypoint.transform

            # Append the x, y, and yaw to the path lists
            self.path['x'].append(transform.location.x)
            self.path['y'].append(transform.location.y)
            self.path['yaw'].append(transform.rotation.yaw)

            # Keep trying to find the next waypoint until a valid one is located
            next_waypoints = current_waypoint.next(2.0)
            if not next_waypoints:
                # print("No further waypoints available.")
                break  # Stop completely if no further waypoints

            # Iterate over potential next waypoints
            valid_waypoint_found = False
            for next_waypoint in next_waypoints:
                # Calculate yaw difference
                yaw_difference = abs(
                    next_waypoint.transform.rotation.yaw - initial_yaw)
                # Normalize within [0, 180]
                yaw_difference = min(yaw_difference, 360 - yaw_difference)

                # Check if yaw difference is within the threshold
                if yaw_difference <= 15:
                    current_waypoint = next_waypoint
                    valid_waypoint_found = True
                    break  # Valid waypoint found, break inner loop

            if not valid_waypoint_found:
                # print(f"All waypoints deviate by more than 30°. Exiting...")
                break

        print(f" ====== Number of waypoints constructed for episode = {
              len(self.path['x'])} =========")

    def _calculate_error_metrics(self):
        self._errors['velocity_error'] = self._set_velocity - self.obs[3]
        veh_tf = self.vehicle.get_transform()
        loc, rot = veh_tf.location, veh_tf.rotation
        # print(len(self.path['x']))
        print(f"self.index  = {self._index}")
        if self._index >= len(self.path['x']):
            return
        cur = np.array([loc.x, loc.y, rot.yaw])
        p1 = np.array([
            self.path['x'][self._index - 1],
            self.path['y'][self._index - 1],
            self.path['yaw'][self._index - 1]
        ])
        p2 = np.array([
            self.path['x'][self._index],
            self.path['y'][self._index],
            self.path['yaw'][self._index]
        ])
        l2 = np.sum((p1[:-1] - p2[:-1]) ** 2)
        scale = np.sum((cur[:-1] - p1[:-1]) * (p2[:-1] - p1[:-1])) / l2
        if scale > 1:
            self._index += 1
        if self._index > 1 and scale < 0:  # making a change here, original was scale < 0
            self._index -= 1
        ref_yaw = p2[2]
        ref_x, ref_y, _ = p1 + max(0, min(1, scale)) * (p2 - p1)
        # foot of perpendicular
        ct_err = np.sqrt(
            (np.subtract([ref_x, ref_y], [loc.x, loc.y]) ** 2).sum())
        direction = []
        for i in [1, -1]:
            angle = np.radians(ref_yaw) + i * np.pi / 2
            e_x = loc.x - ref_x + ct_err * np.cos(angle)
            e_y = loc.y - ref_y + ct_err * np.sin(angle)
            direction.append(e_x ** 2 + e_y ** 2)
        if direction[0] > direction[1]:
            ct_err *= -1
        self._errors['deviation_error'] = ct_err

        ha_err = ref_yaw - rot.yaw
        if ha_err > 180:
            ha_err -= 360
        elif ha_err < -180:
            ha_err += 360
        self._errors['heading_error'] = np.radians(ha_err)


class LossLoggingCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        # Check if it's time to log
        if self.n_calls % self.log_interval == 0:
            if "policy_loss" in self.locals["infos"][0]:
                policy_loss = self.locals["infos"][0]["policy_loss"]
                value_loss = self.locals["infos"][0]["value_loss"]
                print(f"Step {self.n_calls}: Policy Loss = {
                      policy_loss:.4f}, Value Loss = {value_loss:.4f}")
            else:
                print(f"Step {self.n_calls}: Loss information not available")
        return True


if __name__ == "__main__":
    NUM_TIME_STEPS = 50000
    # LOG_INTERVAL = 100
    env = PathFollower(render_mode='human')
    anim = FuncAnimation(plt.gcf(), env.render)
    env.reset()
    # plt.show()

    # check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    print('Training Started ...')

    # # Train the model
    model.learn(total_timesteps=NUM_TIME_STEPS)  # , log_interval=LOG_INTERVAL)

    model.save(f"carla_ppo_{NUM_TIME_STEPS}_{PORT}")

    obs, info = env.reset()
    start, sim_time = time(), 6000

    while time() - start < sim_time:
        # Ask user for input
        # try:
        #     throttle = float(input("Enter throttle value (0 to 1): "))
        #     steer = float(input("Enter steer value (-1 to 1): "))
        #     action = np.array([throttle, steer])
        # except ValueError:
        #     print("Invalid input. Please enter numeric values for throttle and steer.")
        #     continue
        action, _state = model.predict(obs, deterministic=True)
        print(
            f"Action taken - Throttle: {action[0]:.2f}, Steer: {action[1]:.2f}")

        obs, reward, terminated, truncated, info = env.step(action)

        location = env.vehicle.get_transform().location
        print(
            f"Vehicle Location - x: {location.x:.2f}, y: {location.y:.2f}, z: {location.z:.2f}")
        print(f"Vehicle Velocity: {obs[3]:.2f} m/s")
        print(f"Yaw: {env.vehicle.get_transform().rotation.yaw:.2f}")
        print(f"Reward: {reward:.3f}")

        # Save the current frame from the CARLA camera
        # image = env.image_queue.get()
        # image.save_to_disk("/home/user1/Downloads/CARLA_0.9.11/PythonAPI/examples/swarup/test_images/current.png")

        # sleep(1.5)

        if terminated or truncated:
            path_coverage = env._calculate_path_coverage()
            print(f"Path Coverage: {path_coverage:.2f}%")
            obs, info = env.reset()
            print('\n-----RESET-------\n')

    env.close()
    plt.show()

    # RL techniques -> obs scaling -> other training algo
    # MARL
