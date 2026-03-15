import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камера не открылась")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать кадр")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
