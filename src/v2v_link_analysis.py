import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

FREQ_HZ = 5.9e9
TX_POWER_DBM = 23.0
RX_SENSITIVITY_DBM = -85.0
C_LIGHT = 3e8

PL_EXPONENT = 2.6
SHADOWING_STD_DB = 4.0
D0 = 1.0

PL_D0 = 20 * np.log10(4 * np.pi * D0 * FREQ_HZ / C_LIGHT)

FRAME_INTERVAL = 0.1
HYSTERESIS_SECONDS = 0.5
HYSTERESIS_FRAMES = int(HYSTERESIS_SECONDS / FRAME_INTERVAL)
MAP_EDGE_BUFFER_FT = 60.0

LATENCY_SIGMA = 0.4

SDN_MEAN_TIMEOUTS = [0.030, 0.075, 0.200, 0.500, 1.000, 2.000]
COMMUNICATION_RANGES_M = {
    "150m": 150.0,
    "300m": 300.0,
}

def load_and_prep_ngsim(file_path):
    print("\nLoading data...")
    try:
        df = pd.read_csv(
            file_path,
            usecols=["Vehicle_ID", "Global_Time", "Global_X", "Global_Y"],
            engine="c"
        )
    except ValueError:
        print("Missing required columns.")
        sys.exit(1)

    df = df.dropna()
    df = df[(df["Global_X"] > 0) & (df["Global_Y"] > 0)]
    df = df.sort_values("Global_Time")

    df["Global_X_m"] = df["Global_X"] * 0.3048
    df["Global_Y_m"] = df["Global_Y"] * 0.3048

    x_min, x_max = df["Global_X"].min(), df["Global_X"].max()
    y_min, y_max = df["Global_Y"].min(), df["Global_Y"].max()
    
    return df, x_min, x_max, y_min, y_max

def calculate_received_power(distance_meters, shadowing_db):
    distance_meters = np.maximum(distance_meters, 0.1)
    #FSPL
    deterministic_path_loss = PL_D0 + 10 * PL_EXPONENT * np.log10(distance_meters / D0)
    
    path_loss = deterministic_path_loss + shadowing_db
    rx_power = TX_POWER_DBM - path_loss
    return rx_power

def analyze_links(df, x_min, x_max, y_min, y_max, max_range_m):
    print("Running analysis...")
    frames_by_time = df.groupby("Global_Time")

    active_links = {}
    fading_links = {}
    last_x_position = {}
    last_y_position = {}
    # One shadowing draw per pair, reused across frames.
    shadowing_map = {}

    durations = []
    censored = 0
    total_breaks = 0

    for timestamp, frame_data in frames_by_time:
        vehicle_ids = frame_data["Vehicle_ID"].values
        x_meters = frame_data["Global_X_m"].values
        y_meters = frame_data["Global_Y_m"].values
        
        x_feet = frame_data["Global_X"].values
        y_feet = frame_data["Global_Y"].values
        
        last_x_position.update(dict(zip(vehicle_ids, x_feet)))
        last_y_position.update(dict(zip(vehicle_ids, y_feet)))

        coordinates = np.column_stack((x_meters, y_meters))
        
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        
        row_indices, col_indices = np.triu_indices(len(vehicle_ids), k=1)
        pair_distances_m = np.sqrt(squared_distances[row_indices, col_indices])
        
        connected_pairs = set()

        for row_index, col_index, distance_m in zip(row_indices, col_indices, pair_distances_m):
            if distance_m > max_range_m:
                continue

            pair_key = tuple(sorted((vehicle_ids[row_index], vehicle_ids[col_index])))

            if pair_key not in shadowing_map:
                shadowing_map[pair_key] = np.random.normal(0, SHADOWING_STD_DB)

            rx_power = calculate_received_power(distance_m, shadowing_map[pair_key])
            if rx_power >= RX_SENSITIVITY_DBM:
                connected_pairs.add(pair_key)

        # Start tracking new links in this frame.
        for pair_key in connected_pairs - active_links.keys():
            active_links[pair_key] = timestamp
            fading_links.pop(pair_key, None)

        for pair_key in connected_pairs & fading_links.keys():
            fading_links.pop(pair_key)

        for pair_key in list(active_links.keys() - connected_pairs):
            fading_links[pair_key] = fading_links.get(pair_key, 0) + 1

            if fading_links[pair_key] > HYSTERESIS_FRAMES:
                start_time = active_links.pop(pair_key)
                fading_links.pop(pair_key)
                total_breaks += 1

                first_vehicle_y = last_y_position.get(pair_key[0], -1e9)
                second_vehicle_y = last_y_position.get(pair_key[1], -1e9)
                first_vehicle_x = last_x_position.get(pair_key[0], -1e9)
                second_vehicle_x = last_x_position.get(pair_key[1], -1e9)

                first_vehicle_at_edge = (
                    first_vehicle_y < y_min + MAP_EDGE_BUFFER_FT
                    or first_vehicle_y > y_max - MAP_EDGE_BUFFER_FT
                    or first_vehicle_x < x_min + MAP_EDGE_BUFFER_FT
                    or first_vehicle_x > x_max - MAP_EDGE_BUFFER_FT
                )
                second_vehicle_at_edge = (
                    second_vehicle_y < y_min + MAP_EDGE_BUFFER_FT
                    or second_vehicle_y > y_max - MAP_EDGE_BUFFER_FT
                    or second_vehicle_x < x_min + MAP_EDGE_BUFFER_FT
                    or second_vehicle_x > x_max - MAP_EDGE_BUFFER_FT
                )

                if first_vehicle_at_edge or second_vehicle_at_edge:
                    # Edge events are treated as censored observations.
                    censored += 1
                else:
                    duration_ms = timestamp - start_time
                    duration_sec = (duration_ms / 1000.0) - HYSTERESIS_SECONDS
                    if duration_sec >= FRAME_INTERVAL:
                        durations.append(duration_sec)

    # Links still alive at trace end are right-censored.
    censored += len(active_links)

    return np.array(durations), censored, total_breaks

def report_results(durations, total_breaks, censored_count):
    valid_links = len(durations)

    print("\n" + "=" * 70)
    print(" Results")
    print("=" * 70)
    print(f"Total link breaks detected: {total_breaks}")
    print(f"Valid links for analysis: {valid_links}")
    print(f"Censored links: {censored_count}")

    if valid_links == 0:
        return

    print(f"\n{'Mean Latency (s)':<18} | {'Failures (Avg)':<15} | {'P(fail)':<10}")
    print("-" * 55)

    np.random.seed(42)
    
    for mean_lat in SDN_MEAN_TIMEOUTS:
        # Convert target mean to lognormal mu with fixed sigma.
        mu = np.log(mean_lat) - 0.5 * LATENCY_SIGMA**2
        random_latencies = np.random.lognormal(mean=mu, sigma=LATENCY_SIGMA, size=valid_links)
        
        failures = np.sum(durations < random_latencies)
        prob = failures / valid_links
        
        print(f"{mean_lat:<18.3f} | {failures:<15} | {prob:<10.2%}")

def save_plot(durations, base_name):
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=80, density=True, edgecolor='black', alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Link Duration (seconds)")
    plt.ylabel("Probability Density")
    plt.title("V2V Link Duration (Log-Normal Shadowing Model)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = f"{base_name}_PHY_Graph.png"
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")

if __name__ == "__main__":
    FILE_PATH = "<replace with path>"
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)
        
    base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    
    df, xmin, xmax, ymin, ymax = load_and_prep_ngsim(FILE_PATH)
    
    for range_label, max_range_m in COMMUNICATION_RANGES_M.items():
        print(f"\nRange: {range_label}")
        durations, censored_links, total_breaks = analyze_links(
            df,
            xmin,
            xmax,
            ymin,
            ymax,
            max_range_m,
        )
        report_results(durations, total_breaks, censored_links)
        save_plot(durations, f"{base_name}_{range_label}")