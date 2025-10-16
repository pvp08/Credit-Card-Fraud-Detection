from src.fraud_detector import FraudDetector

if __name__ == "__main__":
    print("\nStarting Credit Card Fraud Detection Project...\n")

    detector = FraudDetector('data/creditcard.csv')
    detector.load_data().prepare_data().train_models()
    comparison_df = detector.compare_models()
    detector.save_models()

    print("\nTraining complete! Results:")
    print(comparison_df)
