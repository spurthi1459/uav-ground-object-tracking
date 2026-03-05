import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle
    # cv2.rectangle(frame, top-left corner, bottom-right corner, color BGR, thickness)
    cv2.rectangle(frame, (350, 50), (620, 400), (0, 0, 255), 2)

    # Draw a label above the box
    # cv2.putText(frame, text, position, font, size, color, thickness)
    cv2.putText(frame, "person 0.91", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw a dot at the center of the box
    cv2.circle(frame, (200, 200), 5, (0, 255, 255), -1)  # -1 = filled circle

    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

