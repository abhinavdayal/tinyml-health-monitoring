import numpy as np

class TyrannosaurusOptimizer:
    def __init__(self, func, dim, max_iter=100, population_size=50, alpha=0.5, beta=0.5):
        self.func = func  # Objective function
        self.dim = dim  # Dimension of the search space
        self.max_iter = max_iter  # Maximum number of iterations
        self.population_size = population_size  # Number of individuals in the population
        self.alpha = alpha  # Attraction coefficient
        self.beta = beta  # Repulsion coefficient

    def initialize_population(self):
        return np.random.uniform(-5, 5, (self.population_size, self.dim))

    def optimize(self):
        population = self.initialize_population()
        for _ in range(self.max_iter):
            fitness = [self.func(individual) for individual in population]
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            for i in range(self.population_size):
                if i != best_idx:
                    displacement = self.alpha * (best_individual - population[i]) + self.beta * np.random.uniform(-1, 1, self.dim)
                    population[i] += displacement

        best_fitness = min(fitness)
        best_solution = population[best_idx]
        return best_solution, best_fitness

# Example usage:
def objective_function(x):
    return sum(x**2)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import Model
import random

def RFO_fitness(ESD_Feature, ESD_Label):
  
  sigma_val = []
  learning_rate_val=[]
  accuracy=[]
  for i in range(100):
    sigma = random.uniform(0, 0.001)
    sigma_val .append(sigma)
    learning_rate = random.uniform(0, 0.001)
    learning_rate_val .append(learning_rate)
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(178, 1)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=learning_rate_val[i]), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(ESD_Feature, ESD_Label, epochs=5, batch_size=64)
    model.summary()
    model_feat = Model(inputs=model.input,outputs=model.get_layer('dense').output)
    ESD_NASnet_Features = model_feat.predict(ESD_Feature)
    
    
  return ESD_NASnet_Features
