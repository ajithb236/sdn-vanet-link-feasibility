import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

FREQ_HZ = 5.9e9
TX_POWER_DBM = 23.0
RX_SENSITIVITY_DBM = -85.0
C_LIGHT = 3e8
PL_EXPONENT = 2.6
SHADOWING_STD_DB = 4.0
D0 = 1.0

PL_D0 = 20 * np.log10(4 * np.pi * D0 * FREQ_HZ / C_LIGHT)

LATENCY_SIGMA = 0.4
CTRL_MEANS = [0.03, 0.075, 0.2, 0.5, 1.0, 2.0]

FRAME_INTERVAL = 0.1
HYSTERESIS_SECONDS = 0.5
HYSTERESIS_FRAMES = int(HYSTERESIS_SECONDS / FRAME_INTERVAL)

MAP_EDGE_BUFFER_FT = 60.0

RANGES_M = {
    "150m": 150.0,
    "300m": 300.0
}


def load_ngsim(file_path):
    df = pd.read_csv(
        file_path,
        usecols=["Vehicle_ID", "Global_Time", "Global_X", "Global_Y"]
    ).dropna()

    df = df.sort_values("Global_Time")

    df["X_m"] = df["Global_X"] * 0.3048
    df["Y_m"] = df["Global_Y"] * 0.3048

    x_min, x_max = df["Global_X"].min(), df["Global_X"].max()
    y_min, y_max = df["Global_Y"].min(), df["Global_Y"].max()

    return df, x_min, x_max, y_min, y_max


def rx_power(dist_m, shadow_db):
    dist_m = np.maximum(dist_m, 0.5)
    pl = PL_D0 + 10 * PL_EXPONENT * np.log10(dist_m / D0) + shadow_db
    return TX_POWER_DBM - pl


def analyze_links(df, x_min, x_max, y_min, y_max, R_m):
    grouped = df.groupby("Global_Time")

    active = {}
    fading = {}
    shadowing = {}

    durations = []
    censored = 0
    total = 0

    for t, frame in grouped:
        ids = frame["Vehicle_ID"].values
        xs = frame["X_m"].values
        ys = frame["Y_m"].values

        xs_ft = frame["Global_X"].values
        ys_ft = frame["Global_Y"].values

        coords = np.column_stack((xs, ys))

        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

        iu, ju = np.triu_indices(len(ids), k=1)
        dists = dist[iu, ju]

        pairs = []

        for i, j, d in zip(iu, ju, dists):
            if d > R_m:
                continue

            p = tuple(sorted((ids[i], ids[j])))

            if p not in shadowing:
                shadowing[p] = np.random.normal(0, SHADOWING_STD_DB)

            if rx_power(d, shadowing[p]) >= RX_SENSITIVITY_DBM:
                pairs.append(p)

        pairs = set(pairs)

        for p in pairs - active.keys():
            active[p] = t
            fading.pop(p, None)

        for p in pairs & fading.keys():
            fading.pop(p)

        for p in list(active.keys() - pairs):
            fading[p] = fading.get(p, 0) + 1

            if fading[p] > HYSTERESIS_FRAMES:
                start = active.pop(p)
                fading.pop(p)

                total += 1

                x_u = df.loc[df.Vehicle_ID == p[0], "Global_X"].iloc[-1]
                y_u = df.loc[df.Vehicle_ID == p[0], "Global_Y"].iloc[-1]
                x_v = df.loc[df.Vehicle_ID == p[1], "Global_X"].iloc[-1]
                y_v = df.loc[df.Vehicle_ID == p[1], "Global_Y"].iloc[-1]

                edge = (
                    x_u < x_min + MAP_EDGE_BUFFER_FT or
                    x_u > x_max - MAP_EDGE_BUFFER_FT or
                    y_u < y_min + MAP_EDGE_BUFFER_FT or
                    y_u > y_max - MAP_EDGE_BUFFER_FT or
                    x_v < x_min + MAP_EDGE_BUFFER_FT or
                    x_v > x_max - MAP_EDGE_BUFFER_FT or
                    y_v < y_min + MAP_EDGE_BUFFER_FT or
                    y_v > y_max - MAP_EDGE_BUFFER_FT
                )

                if edge:
                    censored += 1
                else:
                    dur = (t - start) / 1000.0 - HYSTERESIS_SECONDS

                if dur >= FRAME_INTERVAL:
                        durations.append(dur)

    return np.array(durations), total, censored


def feasibility(durations):
    print("\nController Delay | P(fail)")
    print("-----------------------------")

    for mean in CTRL_MEANS:
        mu = np.log(mean) - 0.5 * LATENCY_SIGMA**2

        rng = np.random.default_rng(seed=int(mean * 1e6))
        ctrl = rng.lognormal(mu, LATENCY_SIGMA, size=len(durations))

        pfail = np.mean(durations < ctrl)

        print(f"{mean:>6.3f} s | {pfail:>6.2%}")


def plot(durations, label):
    plt.figure(figsize=(8, 5))

    plt.hist(durations, bins=80, density=True)

    plt.yscale("log")
    plt.xlabel("Link Duration (s)")
    plt.ylabel("PDF (log)")
    plt.title(f"V2V Link Duration ({label}, PHY-aware)")

    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(f"{label}_phy_links.png", dpi=300)

    plt.close()


def main():
    FILE = "lankershim.csv"

    df, xmin, xmax, ymin, ymax = load_ngsim(FILE)

    for label, R in RANGES_M.items():
        print(f"\n=== {label} ===")

        durs, total, cens = analyze_links(df, xmin, xmax, ymin, ymax, R)

        print(f"Valid links: {len(durs)} | Censored: {cens}")

        feasibility(durs)

        plot(durs, label)


if __name__ == "__main__":
    main()