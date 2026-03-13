# VANET Link Lifetime Analysis

This repository contains code for extracting vehicle-to-vehicle (V2V) link lifetimes from vehicular mobility traces. The analysis estimates how long wireless links remain usable between moving vehicles under a simple physical layer model.

Vehicle trajectories are taken from publicly available mobility datasets. Pairwise vehicle distances are computed at each time step, and received signal strength is estimated using a path-loss model. Links are considered active while the received signal remains above a receiver sensitivity threshold.

The script records link durations and evaluates whether these links remain active long enough for a centralized control plane to react.

## Datasets

The analysis uses vehicle trajectory data from:
* NGSIM US-101 freeway dataset
* NGSIM Lankershim Boulevard urban dataset
* LuST (Luxembourg SUMO Traffic) urban mobility scenario

These datasets contain vehicle positions recorded at a sampling rate of 10 Hz.

The datasets are not included in this repository. Place the CSV files in the `data` directory before running the script.

Example filenames:

```
us101.csv
lankershim.csv
lust.csv
```

## Method

The script performs the following steps:

1. Load vehicle trajectory data.
2. Convert position coordinates from feet to meters.
3. Compute pairwise distances between vehicles at each time frame.
4. Estimate received power using a log-distance path loss model.
5. Declare a link active when received power exceeds receiver sensitivity.
6. Track the time interval during which each vehicle pair remains connected.
7. Compute the duration of each link.
8. Compare link durations with several fixed controller delay values.

The output includes link duration distributions and simple failure statistics.

## Assumptions

The analysis uses a simplified wireless communication model.

* Communication occurs at 5.9 GHz.
* Transmission power is fixed.
* Received power is computed using a log-distance path loss model.
* Log-normal shadowing is applied per vehicle pair.
* Fast fading and interference are not modeled.
* Connectivity is determined using a receiver sensitivity threshold.
* Links are evaluated for communication ranges of 150 m and 300 m.
* Short disconnections are filtered using a small hysteresis interval.
* Links that terminate near the spatial boundaries of the dataset are discarded to avoid artifacts from vehicles entering or leaving the observation region.

These assumptions simplify the analysis and should be interpreted as an approximation of real vehicular communication conditions.

## Running the Code

Install dependencies:

```
pip install -r requirements.txt
```

Run the analysis:

```
python src/v2v_link_analysis.py
```

Generated plots will be written to the results directory.

## Dependencies

* Python 3
* numpy
* pandas
* matplotlib

## Notes

This repository focuses on extracting link lifetimes from mobility traces using a simplified physical layer model. It does not implement a full vehicular network simulation or protocol stack.
