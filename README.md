# Carla-rl-term-project

This project simulates an ego vehicle in the CARLA environment using Policy Gradient algorithms. The primary objective is to train the vehicle to follow a predefined path while avoiding collisions and minimizing errors.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Training](#training)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/carla-rl-term-project.git
    cd carla-rl-term-project
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have CARLA installed and running. You can download CARLA 0.9.11 from the [official website](https://carla.org/).

## Usage

1. Start the CARLA server:

    ```sh
    ./CarlaUE4.sh -carla-port = N 
    ```

2. Run the path follower script:
   Modify the PORT variable according to server port.

    ```sh
    python path_follower_single_car_PPO.py
    ```

3. To visualize the path coverage:

    ```sh
    python plot_path_coverage.py
    ```

## Key Components

### PathFollower Class

The `PathFollower` class, defined in [`path_follower_single_car_PPO.py`](path_follower_single_car_PPO.py), is a custom environment that inherits from `gym.Env`. It includes methods for:

- Spawning the ego vehicle and sensors.
- Updating the vehicle's observations.
- Calculating error metrics.
- Creating a global path.
- Rendering the environment.

## Training

To train the model, run the `path_follower_single_car_PPO.py` script. The training process uses the PPO algorithm from Stable Baselines3.

## Report

A Slide deck detailing our project is here [Carla_project](https://docs.google.com/presentation/d/1MZAcfWyvwZUJmLB7GDXJsITIpzXlAZfGfI5yXd8f-pM/edit#slide=id.g31ae0f21128_0_140)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

