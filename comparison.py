"""Compare baseline and ML-assisted RTT, loss, and throughput."""

import matplotlib.pyplot as plt

from ml_model import create_model
from simulation_utils import FEATURE_COLUMNS, OUTPUT_DIR, load_dataset, run_simulation


def main():
    print("Stage 4/4: Comparing both approaches with identical network settings...")
    data = load_dataset()
    model = create_model()
    model.fit(data[FEATURE_COLUMNS], data["Label"])

    baseline = run_simulation()
    ml_assisted = run_simulation(model)

    print("\nSimulation results")
    print(f"{'Metric':<24}{'Baseline':>14}{'ML-assisted':>16}")
    print(
        f"{'Packet loss':<24}{baseline['loss_count']:>14}"
        f"{ml_assisted['loss_count']:>16}"
    )
    print(
        f"{'Throughput (packets/step)':<24}{baseline['throughput']:>14.3f}"
        f"{ml_assisted['throughput']:>16.3f}"
    )
    print(
        f"{'Mean RTT (time units)':<24}"
        f"{sum(baseline['rtt']) / len(baseline['rtt']):>14.3f}"
        f"{sum(ml_assisted['rtt']) / len(ml_assisted['rtt']):>16.3f}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "rtt_comparison.png"
    plt.figure(figsize=(10, 5))
    plt.plot(baseline["rtt"], label="Baseline AIMD")
    plt.plot(ml_assisted["rtt"], label="ML-assisted control")
    plt.title("RTT Comparison: Baseline vs ML-Assisted Simulation")
    plt.xlabel("Simulation step")
    plt.ylabel("RTT (simulation time units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison graph saved to {plot_path.relative_to(OUTPUT_DIR.parent)}.")
    print(
        "The ML model classifies the current congestion state from synthetic "
        "simulation data; this is not a future-event prediction."
    )
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
