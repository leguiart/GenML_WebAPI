import os
import torch
import numpy as np
import sys
import random
from PIL import Image


sys.path.append("../AI/NCA")# Appending repo's root dir in the python path to enable subsequent imports

from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_alpha, to_rgb, make_seed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

CHANNEL_N = 16
CELL_FIRE_RATE = 0.5


isMouseDown = False
running = True

class NCAEnemyGenerator:
    def __init__(self):
        self.eraser_radius = 3
        self.pix_size = 8
        self._map_shape = (72,72)

        self.mid_point = np.array([self.pix_size*self._map_shape[0]/2, self.pix_size*self._map_shape[1]/2])
        self.left_point = self.mid_point + np.array([-self.pix_size*self._map_shape[0]/4, 0.])
        self.right_point = self.mid_point + np.array([self.pix_size*self._map_shape[0]/4, 0.])
        self.up_point = self.mid_point + np.array([0., -self.pix_size*self._map_shape[1]/4])
        self.low_point = self.mid_point + np.array([0., self.pix_size*self._map_shape[1]/4])

        self.max_subdivisions = 15
        self.min_subdivisions = 2

        self.max_intensity = 5.
        self.min_intensity = 0.

        self.slope = (self.min_subdivisions - self.max_subdivisions)/(self.max_intensity - self.min_intensity)

        model_paths = ["./AI/NCA/models/remaster_1.pth"]
        device = torch.device("cpu")

        _rows = np.arange(self._map_shape [0]).repeat(self._map_shape[1]).reshape([self._map_shape[0],self._map_shape[1]])
        _cols = np.arange(self._map_shape[1]).reshape([1,-1]).repeat(self._map_shape[0],axis=0)
        self._map_pos = np.array([_rows,_cols]).transpose([1,2,0])

        self._map = make_seed(self._map_shape , CHANNEL_N)



        self.models = [CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device) for _ in model_paths]
        for i, model_path in enumerate(model_paths): 
            self.models[i].load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.outputs = [model(torch.from_numpy(self._map.reshape([1,self._map_shape[0], self._map_shape[1],CHANNEL_N]).astype(np.float32)), 1) for model in self.models]

        self.disp = displayer(self._map_shape, self.pix_size, rendering=False)

        # Simulate growth
        for i, model in enumerate(self.models):
            frame = 0
            total_frames = 500
            model = self.models[i]
            
            while(frame < total_frames):
                self.outputs[i] = model(self.outputs[i], 1)
                # self._map = to_rgb(self.outputs[i].detach().numpy()[0])
                # self.disp.update(self._map)
                frame += 1

    def _evolve(self, model_index, intensities):
        model = self.models[model_index]
        pixWidth = self.pix_size*self._map_shape[0]
        pixHeight = self.pix_size*self._map_shape[1]

        for intensity in intensities:
            indx = random.randint(0, 4)
            
            if indx == 0:
                initial_point = self.mid_point
                # Can go anywhere
                xOffset = random.uniform(-pixWidth/4, pixWidth/4)
                yOffset = random.uniform(-pixHeight/4, pixHeight/4)
            elif indx == 1:
                initial_point = self.left_point
                # Can go right, up, and down
                xOffset = random.uniform(0., pixWidth/2)
                yOffset = random.uniform(-pixHeight/4, pixHeight/4)
                
            elif indx == 2:
                initial_point = self.right_point
                # Can go left, up and down
                xOffset = random.uniform(-pixWidth/2, 0.)
                yOffset = random.uniform(-pixHeight/4, pixHeight/4)
            elif indx == 3:
                initial_point = self.up_point
                # Can go down, left and right
                xOffset = random.uniform(-pixWidth/4, pixWidth/4)
                yOffset = random.uniform(0., pixHeight/2)
            elif indx == 4:
                initial_point = self.low_point
                # Can go up, left and right
                xOffset = random.uniform(-pixWidth/4, pixWidth/4)
                yOffset = random.uniform(-pixHeight/2, 0.)

            endpoint = initial_point + np.array([xOffset, yOffset])
            vect = ((endpoint - initial_point)/np.linalg.norm(endpoint - initial_point))*1./intensity
            endpoint = initial_point + vect
            

            subdivisions = int(self.slope*intensity + self.max_subdivisions)
            
            positions = np.linspace(initial_point, endpoint, num = subdivisions)

            for pos in positions:
                try:
                    simulated_mouse_pos = np.array([int(pos[1]/self.pix_size), int(pos[0]/self.pix_size)])
                    should_keep = (mat_distance(self._map_pos, simulated_mouse_pos)>self.eraser_radius).reshape([self._map_shape[0],self._map_shape[1],1])
                    self.outputs[model_index] = torch.from_numpy(self.outputs[model_index].detach().numpy()*should_keep)
                except AttributeError:
                    pass
                self.outputs[model_index] = model(self.outputs[model_index], 1)
                
        self._map = to_rgb(self.outputs[model_index].detach().numpy()[0])
        img_arr = self.disp.update(self._map)
        img_arr = img_arr.swapaxes(0,1)

        img = Image.fromarray(img_arr.astype('uint8')).convert('RGBA')
            
        return img

    def Generate(self, **kwargs):
        enemy_id = kwargs['id']
        intensities = kwargs['intensities']
        model_index = enemy_id - 1
        return self._evolve(model_index, intensities)

    
    def GenerateBatch(self, **kwargs):
        enemyTypes = kwargs["enemyTypes"]
        idToImage = {}
        for enemyType in enemyTypes:
            enemy_id = enemyType["id"]
            intensities = enemyType["intensities"]
            idToImage[enemy_id] = self.Generate(id = enemy_id, intensities = intensities)
        return idToImage

        
