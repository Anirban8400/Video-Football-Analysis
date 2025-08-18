from ultralytics import YOLO
model=YOLO(r'D:\Football_Project\models\best.pt')

results=model.predict(r'Data\08fd33_4.mp4', save=True)
print(results[0])
print("_______________________________________")
for box in results[0].boxes:
    print(box)