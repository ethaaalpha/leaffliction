import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=300,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='best_model.pt',
    early_stopping_patience=5
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=3,
                                                           factor=0.5)

    best_val_acc = 0.0
    epochs_without_improvement = 0 

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, hist, labels in tqdm(train_loader,
                                       desc=f"Epoch {epoch+1}/\
                                        {num_epochs} - Train"):
            imgs = [img.to(device) for img in imgs]
            hist = hist.to(device)
            labels = labels.to(device)

            outputs = model(imgs, hist)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, hist, labels in tqdm(val_loader, desc="Validation"):
                imgs = [img.to(device) for img in imgs]
                hist = hist.to(device)
                labels = labels.to(device)

                outputs = model(imgs, hist)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Scheduler] Learning rate updated to {current_lr:.6f}")

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | \
            Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.\n")
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1
            print(f"⏸️ No improvement for \
                {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"⛔ Early stopping triggered. No improvement \
                after {early_stopping_patience} epochs.")
            break

    print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc*100:.2f}%")
