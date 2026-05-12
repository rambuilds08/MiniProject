import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

print("Evaluating XScan AI performance batch-by-batch...")

y_true = []
y_pred = []

# This perfectly matches the images to their labels without the shuffle bug!
for x_batch, y_batch in val_dataset:
    y_true.extend(y_batch.numpy())
    preds = model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

print("\n" + "="*40)
print("🩺 ACCURATE CLINICAL CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('XScan AI - Accurate Diagnostic Confusion Matrix')
plt.ylabel('Actual Doctor Diagnosis')
plt.xlabel('AI Predicted Diagnosis')

plt.savefig('/content/drive/MyDrive/xscan_accurate_confusion_matrix.png', bbox_inches='tight')
print("\n✅ New Confusion Matrix saved as 'xscan_accurate_confusion_matrix.png'")
plt.show()
