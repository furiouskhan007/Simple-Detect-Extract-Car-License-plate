import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

###### Global Variables ######

net = cv2.dnn.readNet("yolov3_training.weights", "yolov3_training.cfg")
classes = ['license']
layers_names = net.getLayerNames()
outputlayers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_license_plate(img):
    if len(img.shape) == 3:
        height, width, channels = img.shape
    else:
        height, width = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    out = net.forward(outputlayers)

    license_plates = []
    for o in out:
        for detect in o:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5 and class_id == 0:  # Assuming '0' corresponds to license plates
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                license_plate = img[y + 2:y + h + 2, x + 2:x + w + 2]
                license_plates.append(license_plate)

    return license_plates


if __name__ == '__main__':
    # Read your image here
    image = cv2.imread("cropped_parking_lot_4_JPG.rf.6ea0a4d2a2672c73e35030e5f8cb513a.jpg")

    # Detect license plates
    plates = detect_license_plate(image)

    # Process each detected license plate
    for i, plate in enumerate(plates):
        # Perform OCR using pytesseract
        plate_text = pytesseract.image_to_string(plate, config='--psm 6')

        # Print the detected text
        print(f"Detected text for license plate {i + 1}: {plate_text}")

        # Display the cropped license plate and its detected text
        cv2.imshow(f"License Plate {i + 1}", plate)
        print(f"Detected text for license plate {i + 1}: {plate_text}")
        cv2.waitKey(0)  # Wait for key press to close the window

    cv2.destroyAllWindows()
