import torch
from sklearn.metrics import classification_report, roc_auc_score
def evaluate_classification(model, test_loader):
    model.eval()
    y_true = torch.tensor([], dtype=torch.long).cuda()
    pred_probs = torch.tensor([]).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        running_loss = 0.0
        for data in test_loader:
            inputs = data["X"].cuda()
            labels = data["y"].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            y_true = torch.cat((y_true, labels), 0)
            pred_probs = torch.cat((pred_probs, outputs), 0)

    # compute predicitions form probs
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(pred_probs, 1)

    y_pred = y_pred.cpu().numpy()
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu().numpy()
    # get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    # macro auc score
    report["auc"] = roc_auc_score(
        y_true, pred_probs, multi_class="ovo", average="macro"
    )
    report["loss"] = running_loss / len(test_loader)
   
    
    return report