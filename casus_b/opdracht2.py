from sklearn.preprocessing import OneHotEncoder
import numpy as np

line = np.array('altijd november altijd regen altijd dit lege hart altijd'.split())
enc = OneHotEncoder()
data = enc.fit_transform(line.reshape(-1, 1))

#gebruik one-hot encoding om te bepalen hoeveel unieke woorden in je corpus zitten
#geef elk woord unieke waarde
#om woorden context te geven gebruik je embeddings, daar zijn getrainde modellen voor
