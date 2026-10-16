"""Train and evaluate a Random Forest congestion-state classifier."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from simulation_utils import FEATURE_COLUMNS, load_dataset


def create_model():
    """Create the project's reproducible Random Forest classifier."""
    return RandomForestClassifier(n_estimators=100, random_state=42)


def main():
    print("Stage 2/4: Training and evaluating the congestion-state classifier...")
    data = load_dataset()
    features = data[FEATURE_COLUMNS]
    labels = data["Label"]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = create_model()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(features_train)}")
    print(f"Test rows: {len(features_test)}")
    print(f"Test accuracy: {accuracy_score(labels_test, predictions):.4f}")
    print(
        "Confusion matrix (rows = actual, columns = predicted; order [0, 1]):\n"
        f"{confusion_matrix(labels_test, predictions, labels=[0, 1])}"
    )
    report = classification_report(
        labels_test,
        predictions,
        labels=[0, 1],
        target_names=["not congested", "congested"],
        zero_division=0,
    )
    print(f"Classification report:\n{report}")
    print(
        "Note: this evaluates congestion-state classification, not prediction "
        "of a future congestion event."
    )


if __name__ == "__main__":
    main()
