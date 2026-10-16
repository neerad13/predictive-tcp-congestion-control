"""Shared settings and queue simulation used by the project scripts."""

from pathlib import Path

import pandas as pd

TIME_STEPS = 5000
QUEUE_CAPACITY = 50
PACKETS_PROCESSED_PER_STEP = 3
BASE_DELAY = 50
DELAY_FACTOR = 2
ADDITIVE_INCREASE = 0.05
MULTIPLICATIVE_DECREASE = 0.5
ML_RATE_MULTIPLIER = 0.7
ML_QUEUE_THRESHOLD = 0.6
CONGESTION_LABEL_THRESHOLD = 0.8

FEATURE_COLUMNS = ["RTT", "RTT_Change", "Queue_Size", "Sending_Rate"]
DATA_COLUMNS = FEATURE_COLUMNS + ["Label"]
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "network_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_dataset():
    """Read the generated dataset and check its expected column order."""
    data = pd.read_csv(DATASET_PATH)
    if list(data.columns) != DATA_COLUMNS:
        expected = ", ".join(DATA_COLUMNS)
        actual = ", ".join(data.columns)
        raise ValueError(
            f"Unexpected dataset columns. Expected [{expected}], got [{actual}]. "
            "Run network_simulation.py to regenerate the dataset."
        )
    return data


def run_simulation(model=None):
    """Run baseline AIMD or model-assisted control and return samples and metrics.

    The model, when supplied, classifies the current observed congestion state.
    Features are recorded before the rate-control decision at each step.
    """
    sending_rate = 1.0
    queue_size = 0
    previous_rtt = BASE_DELAY
    loss_count = 0
    throughput_count = 0
    samples = []
    rtt_values = []

    for _ in range(TIME_STEPS):
        packets_sent = int(sending_rate)
        packet_dropped = False

        for _ in range(packets_sent):
            if queue_size < QUEUE_CAPACITY:
                queue_size += 1
            else:
                loss_count += 1
                packet_dropped = True

        processed = min(queue_size, PACKETS_PROCESSED_PER_STEP)
        queue_size -= processed
        throughput_count += processed

        rtt = BASE_DELAY + (queue_size * DELAY_FACTOR)
        rtt_change = rtt - previous_rtt
        previous_rtt = rtt

        label = int(queue_size > CONGESTION_LABEL_THRESHOLD * QUEUE_CAPACITY)
        samples.append([rtt, rtt_change, queue_size, sending_rate, label])
        rtt_values.append(rtt)

        if model is None:
            if packet_dropped:
                sending_rate *= MULTIPLICATIVE_DECREASE
            else:
                sending_rate += ADDITIVE_INCREASE
        else:
            features = pd.DataFrame(
                [[rtt, rtt_change, queue_size, sending_rate]],
                columns=FEATURE_COLUMNS,
            )
            prediction = model.predict(features)[0]
            if prediction == 1 and queue_size > ML_QUEUE_THRESHOLD * QUEUE_CAPACITY:
                sending_rate *= ML_RATE_MULTIPLIER
            else:
                sending_rate += ADDITIVE_INCREASE

        sending_rate = max(sending_rate, 1.0)

    return {
        "data": pd.DataFrame(samples, columns=DATA_COLUMNS),
        "rtt": rtt_values,
        "loss_count": loss_count,
        "throughput": throughput_count / TIME_STEPS,
    }
