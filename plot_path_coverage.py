import re
import matplotlib.pyplot as plt

# Function to extract data


def extract_path_coverage(file_path):
    training_episodes = []
    path_coverage_values = []

    # Open and process the file
    with open(file_path, 'r') as file:
        episode_count = 0
        for line in file:
            if "###" in line and "Path Coverage" in line:
                # Extract the path coverage value
                coverage_match = re.search(r"Path Coverage:\s*([\d.]+)%", line)
                if coverage_match:
                    coverage = float(coverage_match.group(1))
                    episode_count += 1  # Increment episode count
                    # Track episode number
                    training_episodes.append(episode_count)
                    # Track path coverage value
                    path_coverage_values.append(coverage)

    return training_episodes, path_coverage_values

# Function to plot the data


def plot_path_coverage(training_episodes, path_coverage_values):
    plt.figure(figsize=(10, 6))
    plt.plot(training_episodes, path_coverage_values,
             marker='o', label='Path Coverage')
    plt.title("Path Coverage vs Training Episode")
    plt.xlabel("Training Episode")
    plt.ylabel("Path Coverage (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("path.png")


# Example usage
file_path = "carla_ppo_3.log"  # Replace with your file path
episodes, coverage = extract_path_coverage(file_path)

if episodes and coverage:
    plot_path_coverage(episodes, coverage)
else:
    print("No valid Path Coverage data found in the file.")
