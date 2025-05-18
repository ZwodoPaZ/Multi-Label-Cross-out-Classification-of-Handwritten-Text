if __name__ == "__main__":
    
    """
    Disclaimer:
    The code was not originally run in this configuration.
    It has been tested in the current configuration but not extensivly.
    Running the current configuration you might encounter issues.
    """
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from dataset_dataloaders import train_binary_loader, val_binary_loader, test_binary_loader, BATCH_SIZE, train_binary, val_binary
    from training import train_model
    from model_implementations import ResidualAttentionModel

    # - - - - - Device setting - - - - - #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # - - - - - Hyperparameters - - - - - #
    LEARNING_RATE = 1e-5
    EPOCHS = 100
    model = ResidualAttentionModel(num_classes=2)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    trained_model = train_model(
        model, criterion, optimizer, scheduler,
        train_binary_loader, val_binary_loader, EPOCHS,
        BATCH_SIZE, device, len(train_binary), len(val_binary),
        "./models/BinaryClassification", binary=True
    )

    # - - - - - Testing Binary - - - - - #
    model = torch.load("./models/BinaryClassificationFinal", weights_only=False)
    model.to(device)

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    incorrect_indices = []

    with torch.no_grad():
        for test_nr, (images, labels, index) in enumerate(test_binary_loader):
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            pred_classes = torch.max(F.softmax(prediction, dim=1), dim=1)[1]
            correct += torch.sum(pred_classes == labels).item()
            incorrect_indices += [
                (index[x].item(), pred_classes[x].item(), labels[x].item())
                for x in range(len(pred_classes))
                if pred_classes[x] != labels[x]
            ]
            total += labels.size(0)
            for (i, j) in zip(labels, pred_classes):
                if i == 0 and j == 0:
                    true_positive += 1
                elif i == 0 and j == 1:
                    false_positive += 1
                elif i == 1 and j == 0:
                    false_negative += 1
            print(str(test_nr), end="\r")

    print("Total: ", total)
    print("Incorrect: ", total - correct)
    print("Accuracy: ", correct / total)
    print("F1 Score: ", true_positive / (true_positive + 0.5 * (false_positive + false_negative)))