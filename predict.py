#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
re Created on 23.09.2000

@author: maitry
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class Classifier:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        # loading model
        model = load_model('model.h5')

        category = {0: 'Crishtiano Ronaldo', 1: 'Diego Maradona', 2: 'Lionel Messi', 3: 'Kylian Mbappé', 4: 'Ronaldinho Gaúcho'}
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        imge = image.img_to_array(test_image)
        imge = imge.reshape(-1, 64, 64, 3)
        pred = model.predict(imge)
        cls = np.argmax(pred)
        result = category[cls]
        return [result]