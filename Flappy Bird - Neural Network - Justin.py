import pygame
import sys
import random
import math
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
pygame.init()

#Contants
WIDTH = 800
HEIGHT = 600
speed = 3
fps = 60 * speed  #integers only
G = -1
p_spawn_time = 1
#end Constants
background_color = (43,42,39)

#NN goodies
gen_size = 100
mutation_rate = 0.5
##end NN goodies


screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Flappy Bird - Genetic Algorithm -- Justin Stitt")
clock = pygame.time.Clock()
sched = BackgroundScheduler()
sched.start()


birds = []
pipes = []

class Bird():
    def __init__(self):
        self.pos = (50,random.randint(100,500))
        self.size = 30
        self.v = 0
        self.color = (190,43,212) #magenta
        self.jump_force = 15
        self.brain = Brain()
        self.fitness = 0
    def update(self):
        global G
        self.fill_inputs()
        if(self.brain.think() > .5): #THINK
            self.jump()
        self.fitness += 1/len(birds)
        self.v += G
        self.pos = (self.pos[0],self.pos[1] - self.v)
        self.render()
        if(self.collision() == 1): # dead?
            self.die()
    def render(self):
        pygame.draw.ellipse(screen,self.color,(self.pos[0],self.pos[1],self.size,self.size))
    def jump(self):
        self.v = self.jump_force
    def collision(self):
        """returns 1 if dead"""
        if(self.pos[1] > HEIGHT):
            return 1
        elif(self.pos[1] < 0):
            return 1
        #top pipe collision
        for pipe in pipes:
            if(self.pos[0] + self.size > pipe.t_box[0] and self.pos[0]  < pipe.t_box[0] + pipe.t_box[0] + pipe.t_box[2]):
                if(self.pos[1] + self.size > pipe.t_box[1] and self.pos[1] < pipe.t_box[1] + pipe.t_box[3]):
                    return 1
            if(self.pos[0] + self.size > pipe.b_box[0] and self.pos[0]  < pipe.b_box[0] + pipe.b_box[0] + pipe.b_box[2]):
                if(self.pos[1] + self.size > pipe.b_box[1] and self.pos[1]  < pipe.b_box[1] + pipe.b_box[3]):
                    return 1
    def die(self):
        global birds
        if(len(birds) == 1):#Am i the last bird??? if so, pass my neural network weights to the next generation! I am the best birdie ever!
            neural_network.best_net = self.brain.weights
            neural_network.copy()

        birds.remove(self)
    def fill_inputs(self):
        global pipes
        inputs = [0] * 4
        inputs[0] = self.pos[1] /HEIGHT #bird y pos
        inputs[1] = pipes[0].x  /WIDTH#closest pipe x pos
        inputs[2] = pipes[0].t_box[3] /HEIGHT#closest pipe top bounding box
        inputs[3] = pipes[0].b_box[1] /HEIGHT# closest pipe bot bounding box
        self.brain.layers[0] = inputs

class Pipe():
    def __init__(self):
        self.gap = random.randint(75,150)
        self.y = random.randint(200,400)
        self.x = 1000
        self.v = -5
        self.color = (43,243,65)
        self.w = 50
        self.t_box = (0,0,0,0)
        self.b_box = (0,0,0,0)
    def update(self):
        self.x = self.x + self.v
        self.render()
    def render(self):
        h1 = self.y - self.gap
        self.t_box = (self.x,0,self.w,h1)
        self.b_box = (self.x,self.y + self.gap,self.w,HEIGHT - (self.gap + self.y))
        pygame.draw.rect(screen,self.color,self.t_box)#top pipe
        pygame.draw.rect(screen,self.color,(self.x,self.y + self.gap,self.w,HEIGHT - (self.gap + self.y)))#bot pipe

class Brain():
    def __init__(self):
        self.inputs = [0] * 4 # number of inputs
        self.hidden_layers =  [0] * 5  #inner most array are the neuron values, outermost is each individual layer of array of neurons
        self.outputs = [0] * 1 #number of outputs
        self.layers = []
        self.layers.append(self.inputs)
        self.layers.append(self.hidden_layers)
        self.layers.append(self.outputs)
        self.weights = []
        self.init_weights()
    def init_weights(self):
        #THIS WORKS. BUT... If i want more than ONE hidden layer, how do i do create more weights.
        for x in range(1,len(self.layers)):
            neurons_in_layer = len(self.layers[x])
            neurons_in_prev_layer = len(self.layers[x - 1])
            self.weights.append(2 * np.random.random((neurons_in_layer,neurons_in_prev_layer)) - 1)  #(x,y)  x = neurons in layer, y = neurons in previous layer
    def think(self):
        for x in range(1,len(self.layers)): #run 2 times ( 3 layers )
            for y in range(len(self.layers[x]) ):
                self.layers[x][y] = sigmoid(np.dot(self.layers[x-1],self.weights[x-1][y]))
        output = self.layers[-1][0]
        return output
        #input layer times weights


class NeuralNetwork():
    def __init__(self):
        self.best_net = []
    def copy(self):
        global birds
        remove_pipes()
        create_birds(gen_size)
        mutation_rate = 3/birds[0].fitness
        print(mutation_rate)
        for b in range(len(birds)):
            for x in range(len(self.best_net)):
                for y in range(len(self.best_net[x])):
                    for z in range(len(self.best_net[x][y])):
                        birds[b].brain.weights[x][y][z] = self.best_net[x][y][z]#(random.uniform(1-mutation_rate,1+mutation_rate))
                        if(random.uniform(0,1) < mutation_rate):
                            birds[b].brain.weights[x][y][z] += random.triangular(-1, 1)

        #print("-==--==--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-==-=-=--==-=-=-=-=-=-=-=-==--==-=-=-= birds 0 BELOW")
        #print(birds[0].brain.weights)
        #print("-==--==--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-==-=-=--==-=-=-=-=-=-=-=-==--==-=-=-= birds 1 BELOW")
        #print(birds[1].brain.weights)
        #print("-==--==--=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-==-=-=--==-=-=-=-=-=-=-=-==--==-=-=-= birds 2 BELOW")
        #print(birds[2].brain.weights)

neural_network = NeuralNetwork()

def create_birds(num):
    for x in range(num):
        birds.append(Bird())
def remove_pipes():
    for pipe in pipes:
        if pipe.x < WIDTH//2:
            pipes.pop(0)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



create_birds(gen_size)
#birds.append(Bird())
pipes.append(Pipe())

def pipe_spawner():
    pipes.append(Pipe())
sched.add_job(pipe_spawner,'interval',seconds = p_spawn_time/speed)

def exit():
    pygame.quit()
    sys.exit()

def update():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_SPACE:
                if(len(birds) >= 1):
                    birds[0].jump()
                    print(birds[0].fitness)
            if event.key == pygame.K_ESCAPE:
                exit()
    for bird in birds:
        bird.update()
    for pipe in pipes:
        pipe.update()
        if(pipe.x < -pipe.w):
            pipes.remove(pipe)

def render():
    pass

while True:
    screen.fill(background_color)
    update()
    render()
    pygame.display.flip()
    clock.tick(fps)
