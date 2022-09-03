
import random
import numpy as np
from PIL import Image
from sympy import rad, re
from randimage import get_random_image

class BaseRandomImageGenerator:
    def GenerateBatch(self, **kwargs):
        batch_size = kwargs["batch_size"]
        selected = kwargs["selected"] if "selected" in kwargs else None
        img_list = []
        if not selected:
            for _ in range(batch_size):
                img_list += [self.Generate(**kwargs)]
            return img_list
        else:
            offspring = []
            parents = selected
            random.shuffle(parents)
            while len(offspring) < batch_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                if random.random() <= 0.9:
                    c1, c2 = self.Crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                
                if random.random() <= 1./batch_size:
                    c1 = self.Mutate(c1.copy())
                if random.random() <= 1./batch_size:
                    c2 = self.Mutate(c2.copy())
                offspring += [c1, c2]

            return offspring[:batch_size]


    def Crossover(self, img1, img2):
        point = random.random()
        return Image.blend(img1, img2, point), Image.blend(img2, img1, point)

    def Mutate(self, img):
        mut = np.array(img)
        A = mut.shape[0] / (random.random() * 10.)
        w = 5. * random.random() / mut.shape[1]

        shift = lambda x: A * np.sin(2.0*np.pi*x * w)

        for i in range(mut.shape[0]):
            mut[:,i] = np.roll(mut[:,i], int(shift(i)))

        return Image.fromarray(mut.astype('uint8')).convert('RGBA')

    def Generate(self, **kwargs):
        pass
    

class NumpyRandomImageGenerator(BaseRandomImageGenerator):

    def Generate(self, **kwargs) -> Image:
        width = kwargs["width"]
        height = kwargs["height"]
        black_white = kwargs["black_white"]

        imarray = np.random.rand(width,height,3) * 255
        if black_white:
            img = Image.fromarray(imarray.astype('uint8')).convert('L')
        else:
            img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        
        return img

class RandImageGenerator(BaseRandomImageGenerator):

    def Generate(self, **kwargs) -> Image:
        width = kwargs["width"]
        height = kwargs["height"]
        black_white = kwargs["black_white"]

        img_size = (width,height)
        return Image.fromarray(get_random_image(img_size).astype('uint8')) if black_white else Image.fromarray(get_random_image(img_size).astype('uint8'))


        
        

