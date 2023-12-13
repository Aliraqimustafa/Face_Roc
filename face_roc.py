from deepface import DeepFace
from deepface.commons import functions, distance as dst
import cv2
import json
import numpy as np


model_name = 'Facenet512'
detector_backend="yolov8",
distance_metric="cosine",
enforce_detection=False,
align=True,
normalization="base"
target_size = (160 , 160)



def add_img2db(img , label , path_db_vex = 'database_vec.txt'):
  try:
    open(path_db_vex)
  except:
    json.dump({'id_0' : [1 for i in range(262)]}, open(path_db_vex,'w'))
  db = json.load(open(path_db_vex))
  n_id = list(db.keys())[-1].split('_')[1]
  n_id = int(n_id) + 1
  key = f'id_{n_id}'
  db[key] = (DeepFace.represent(img_path = img , model_name = model_name,
                                enforce_detection=enforce_detection)[0]["embedding"] ,label )
  json.dump(db, open(path_db_vex,'w'))
  print('"add_img2db" has completed with id: {} and label : {}'.format(key , label))

def get_db(path_db_vex = 'database_vec.txt'):
  db = json.load(open(path_db_vex))
  return db

def cheak_content(content):
  v1 = DeepFace.represent(content , model_name = model_name , enforce_detection=False)[0]["embedding"]
  distances = []
  db = list(get_db().values())[1:]
  for v2 in db :
      distance = dst.findCosineDistance(v1, v2[0])
      distances.append(distance)
  distance = min(distances)
  index = np.argmin(distances)
  threshold = dst.findThreshold(model_name, 'cosine')
  # threshold = 0.5
  print(distance ,threshold )
  verified = distance <= threshold
  if verified:
    name = db[index][1]
  else:
    name = 'UnKnown'
  return {"verified": verified , 'name' : name}

def cheak_1face(img):
  region = {}
  try:
    obj = functions.extract_faces(
        img=img,
        target_size=target_size,
        detector_backend='yolov8',
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,)[0]
    region['img'] = []
    region['img'].append(cheak_content(img))
    region['img'].append(obj[1])
  except:
    region['img'] = []
  return region

def draw_box(img , x,y,w,h , color = (0, 255, 0) , size = 4):
    top_left = (x ,y)
    bottom_right = (x , y + int(h/4))
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (x ,y)
    bottom_right = (x + int(w/4), y)
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (w+x ,y)
    bottom_right = (w+x , y + int(h/4))
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (w+x ,y)
    bottom_right = (w+x - int(x/4), y)
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (x ,y+h)
    bottom_right = (x , y+h - int(h/4))
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (x ,y+h)
    bottom_right = (x + int(w/4), y+h)
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (x+w ,y+h)
    bottom_right = (x+w , y+h - int(h/4))
    img = cv2.line(img, top_left, bottom_right, color, size)
    
    top_left = (x+w ,y+h)
    bottom_right = (x+w - int(w/4), y+h)
    img = cv2.line(img, top_left, bottom_right, color, size)


def draw_on_frame(frame):
    r = cheak_1face(frame)
    if r['img'] ==[]:
        return frame
    coo = r['img'][1]
    x, y, w, h = coo['x'], coo['y'], coo['w'], coo['h']
    if r['img'][0]['verified']:
        color1 = (0,255,0)
    else:
        color1 = (0,0,255)
    draw_box(frame , x,y,w,h , color = color1 , size=7)
    
    
    text = r['img'][0]['name']

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (x - int(x * .1) , y - int(y * .1))
    text_size = 2
    color = (255,0,0)
    frame = cv2.putText(frame, text, text_position, font, text_size, color, 2, cv2.LINE_AA)
    
    return frame


