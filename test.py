from api.v1.endpoints.ImageCaption import *
from api.v1.endpoints.GetTextFeature import *
from api.v1.endpoints.ConnectQdrant import *
from core.config import settings
import random
db = ConnectQdrant(settings.HOST, settings.PORT)
# F = GetTextFeature(settings.W_MODEL_FEATURE)
# feature1 = F.get_feature("She gazed out the window at the rain, feeling a mix of nostalgia and hope for the future.")
# feature2 = F.get_feature(
#     "Looking through the window at the falling rain, she felt a blend of longing and anticipation for what lies ahead.")
# print(feature1)
random_list = [random.random() for _ in range(1024)]
db.insert("testv1",1,random_list)
