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
    from dataset_dataloaders import train_multi_loader, val_multi_loader, test_multi_loader, BATCH_SIZE, train_multi, val_multi
    from model_implementations import ResidualAttentionModel
    from training import train_model

    # - - - - Training Multi Label - - - - - #

    # - - - - - Device setting - - - - - #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # - - - - - Hyperparamters - - - - - #

    LEARNING_RATE = 1e-3
    EPOCHS = 100

    model = ResidualAttentionModel(num_classes = 8)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay = 0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    trained_model = train_model(model, criterion, optimizer, scheduler, 
                            train_multi_loader, val_multi_loader, EPOCHS, 
                            BATCH_SIZE, device, len(train_multi), len(val_multi), "./models/MultiClassification")

    # - - - - - Testing Multi - - - - - #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("./models/MultiClassification", weights_only = False)

    model.to(device)
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    num_classes = 8
    class_bias = []
    class_prevalence = []
    class_correct = []
    F1_macro_score = 0

    for _ in range(num_classes):
        class_bias.append(0)
        class_prevalence.append(0)
        class_correct.append(0)

    with torch.no_grad():
        for test_nr, (images, labels, index) in enumerate(test_multi_loader):
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            #correct += torch.sum((torch.max(F.softmax(prediction, dim=1),dim=1)[1]==labels)).item()
            correct += torch.sum(prediction==labels).item()
            
            total += labels.size(0)
            print(str(test_nr), end = "\r")
            
            for (label, prediction) in zip(labels, prediction):
            #for (label, prediction) in zip(labels, torch.max(F.softmax(prediction, dim=1),dim=1)[1]):
                if label == prediction:
                    class_correct[label] += 1
                else:
                    class_bias[label] += 1       # False negative
                    class_prevalence[prediction] += 1  # False positive
            
        for i in range(num_classes):
            TP = class_correct[i]
            FN = class_bias[i]
            FP = class_prevalence[i]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)
            else:
                F1 = 0

            F1_macro_score += F1

        F1_macro_score /= num_classes

        
    #assert len(incorrect_indices) == total-correct     
    print("Total: ", total)               
    print("Incorrect: ", total-correct)
    print("accuracy: ", correct/total)
    print("F1-Macro: ", F1_macro_score)