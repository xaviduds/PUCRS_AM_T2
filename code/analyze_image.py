from ultralytics import YOLO
import numpy as np
import cv2
import time
from scipy.ndimage import label

model = YOLO("runs/segment/train/weights/best.pt")

def analyze_image(path: str):
  predictions = []
  results = model.predict(path)
  for bottle in results:
    for i, box in enumerate(bottle.boxes):
      if(box.cls == 39):
        predictions.append(get_liquid_level(i, bottle, path))

  return predictions

def get_liquid_level(i, bottle, path):
  begin = time.time()

  resized_mask = cv2.resize(bottle.masks.masks[i].numpy(), (bottle.orig_img.shape[1], bottle.orig_img.shape[0])).astype(bool)
  masked_image = bottle.orig_img * resized_mask[:, :, np.newaxis]

  valid = np.where(np.any(masked_image != 0, axis=-1))
  min_y = np.min(valid[0])
  max_y = np.max(valid[0])
  min_x = np.min(valid[1])
  max_x = np.max(valid[1])

  cropped_image = masked_image[min_y:max_y, min_x:max_x]
  resized_image = cv2.resize(cropped_image, (96, 256)) # resize to a common size of a bottle

  bottle_height = np.shape(resized_image)[0]

  top_section = resized_image[int(bottle_height * 0.12):int(bottle_height * 0.2),:,:]
  top_color = np.array([
      top_section[:,:,0][top_section[:,:,0] != 0].mean(), 
      top_section[:,:,1][top_section[:,:,1] != 0].mean(), 
      top_section[:,:,2][top_section[:,:,2] != 0].mean()
  ])

  bottom_section = resized_image[int(bottle_height * 0.9):int(bottle_height * 0.95),:,:]
  bottom_color = np.array([
      bottom_section[:,:,0][bottom_section[:,:,0] != 0].mean(), 
      bottom_section[:,:,1][bottom_section[:,:,1] != 0].mean(), 
      bottom_section[:,:,2][bottom_section[:,:,2] != 0].mean()
  ])

  distance = np.linalg.norm(bottom_color - top_color) # color difference
  max_distance = 255 * 3 ** (1/2) # maximum difference
  step_size =  255 / max_distance

  if(distance > 30):
    img = np.apply_along_axis(lambda x: abs(np.linalg.norm(bottom_color - x) * step_size - 255) if np.sum(x) != 0 else 0, axis=2, arr=resized_image).astype(int) # expensive function
    img = cv2.blur(src=img, ksize=(7, 7))
    threshold = img > 220
    
    labeled_array, num_features = label(threshold)
    labels, counts = np.unique(labeled_array, return_counts=True)

    counts = counts[1:]
    labels = labels[1:]

    largest_cluster_label = labels[np.argmax(counts)]
    largest_cluster = labeled_array == largest_cluster_label

    valid = np.where(largest_cluster)
    min_y = np.min(valid[0])
    max_y = np.max(valid[0])
    min_x = np.min(valid[1])
    max_x = np.max(valid[1])

    final_image = largest_cluster[min_y:max_y, min_x:max_x]
    liquid_height = np.shape(final_image)[0]

    fill_percentage = int((liquid_height / bottle_height) * 100)

    end = time.time()
    return {
      '%': fill_percentage,
      'path': path,
      'img': cropped_image,
      'duration': end - begin
    }

  else:
    # Bottle is empty or full, lets assume that is empty
    end = time.time()
    return {
      '%': 0,
      'path': path,
      'img': cropped_image,
      'duration': end - begin
    }
