prediction = sclera_detection_model.predict(image , conf = 0.6)

masks = []

for i, box in enumerate(prediction[0].boxes):
	cls_id = int(box.cls[0])  # Class index
	conf = float(box.conf[0])  # Confidence score
	label = prediction[0].names[cls_id]  # Class name
	x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

	if label == 'sclera':
	    mask = prediction[0].masks.data[i]
	    masks.append(mask)

combined_mask = torch.stack(masks).sum(dim=0).clamp(0, 1)
binary_mask = (combined_mask.cpu().numpy() * 255).astype(np.uint8)
binary_mask = cv2.resize(binary_mask , (image.shape[1], image.shape[0]))
binary_mask = cv2.merge([binary_mask]*3)

sclera_image = cv2.bitwise_and(image , binary_mask)
_,sclera_mask = cv2.threshold(cv2.cvtColor(sclera_image , cv2.COLOR_RGB2GRAY) ,0,255 ,cv2.THRESH_OTSU + cv2.THRESH_BINARY)
sclera_image = cv2.bitwise_and(cv2.merge([sclera_mask]*3) , sclera_image)

new_sclera_data = cv2.cvtColor(sclera_image , cv2.COLOR_RGB2HSV)

lower_yellow = np.array([10, 20, 20])   # picks up pale yellow
upper_yellow = np.array([40, 255, 255]) 

yellow_mask = cv2.inRange(new_sclera_data, lower_yellow, upper_yellow)

yellow_count = np.count_nonzero(yellow_mask)
total_sclera = np.count_nonzero(sclera_image)
JI = yellow_count / total_sclera
