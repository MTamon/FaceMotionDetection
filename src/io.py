from gc import garbage
from msilib.schema import Error
import pickle
from typing import List

AREA_KEYS = ["xmin","ymin", "xmax", "ymax", \
    "width", "height", "width_min", "height_min", "width_max", "height_max", \
        "success"]

def write_face_area(path, face_area: List[dict]):
    # check dict keys
    for area in face_area:
        for key in AREA_KEYS:
            if not key in area.keys():
                raise InvalidDictError("dictionary lucks key.")
            
    # output by pickle
    with open(path, 'wb') as f:
        pickle.dump(face_area, f)
        
def load_face_area(path) -> List[dict]:
    with open(path, 'rb') as f:
        face_area = pickle.load(f)
        
    return face_area
    
class InvalidDictError(Exception):
    pass