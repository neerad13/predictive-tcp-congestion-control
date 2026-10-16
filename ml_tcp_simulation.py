"""Run the model-assisted TCP-like simulation using the generated dataset."""

import matplotlib.pyplot as plt

from ml_model import create_model
from simulation_utils import FEATURE_COLUMNS, OUTPUT_DIR, load_dataset, run_simulation


def main():
    print("Stage 3/4: Training the classifier on generated simulation data...")
    data = load_dataset()
    model = create_model()
    model.fit(data[FEATURE_COLUMNS], data["Label"])

    print("Running the ML-assisted TCP-like simulation...")
    result = run_simulation(model)
    print(f"ML-assisted packet loss: {result['loss_count']}")
    print(f"ML-assisted throughput: {result['throughput']:.3f} packets per step")

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "ml_tcp_simulation_rtt.png"
    plt.figure(figsize=(10, 5))
    plt.plot(result["rtt"], label="ML-assisted TCP-like simulation")
    plt.title("Round-Trip Time During ML-Assisted Simulation")
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
