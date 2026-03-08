from src.dataset import LeafImageDataset, make_splits, collect_samples

if __name__ == '__main__':
    samples, labels, _, _ = collect_samples()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(samples, labels)

    dataset = LeafImageDataset(samples, labels)
    print(dataset[1899])
