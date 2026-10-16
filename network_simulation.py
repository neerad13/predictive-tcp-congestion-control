"""Generate a reproducible dataset from the baseline TCP-like simulation."""

import matplotlib.pyplot as plt

from simulation_utils import DATASET_PATH, OUTPUT_DIR, PROJECT_DIR, run_simulation


def main():
    print("Stage 1/4: Running the baseline TCP-like network simulation...")
    result = run_simulation()

    result["data"].to_csv(DATASET_PATH, index=False)
    print(
        f"Dataset saved to {DATASET_PATH.relative_to(PROJECT_DIR)} "
        f"({len(result['data'])} rows)."
    )
    print(f"Baseline packet loss: {result['loss_count']}")
    print(f"Baseline throughput: {result['throughput']:.3f} packets per step")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "network_simulation_rtt.png"
    plt.figure(figsize=(10, 5))
    plt.plot(result["rtt"], label="Baseline TCP-like simulation")
    plt.title("Round-Trip Time During Baseline Simulation")
    plt.xlabel("Simulation step")
    plt.ylabel("RTT (simulation time units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"RTT graph saved to {plot_path.relative_to(OUTPUT_DIR.parent)}.")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
