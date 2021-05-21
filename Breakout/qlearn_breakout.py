#!/usr/bin/env python


#https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import random
import numpy as np
from collections import deque

import json
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD , Adam
import tensorflow as tf
import agent
# Import the gym module
import gym

GAME = 'atari' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training. de cada 3200 frames, vamos ao nosso buffer e selecionamos de forma aleatoria um batch size. Neste caso, 32 frames. em numpy arrays
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon EPSILON é para ver o exploration vs exploitation
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows, img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (8, 8), strides = (4, 4), padding = 'same', input_shape = (img_rows, img_cols, img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr = LEARNING_RATE)
    model.compile(loss='mse', optimizer = adam)
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    env =  gym.make('BreakoutDeterministic-v4')
    env.reset()
    env.render()
    # store the previous observations in replay memory
    D = deque()
    
    #----------------------------------------
    #PARA OBTER O SIGNIFICADO DAS AÇÕES POSSíVEIS
    print(env.unwrapped.get_action_meanings())
    #----------------------------------------
    
    # get the first state by doing nothing and preprocess the image to 80x80x4
        
    x_t, r_0, terminal, info = env.step(1) #COMEÇAR O JOGO COM A AÇÃO "FIRE"
    env.render()

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80,80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range = (0,255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2) #colocar a sequência de frames. 4 frames sequenciais, que vamos aplicar à nossa lista. Para conseguir a estabilidade de imagens sequenciais
    
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4



    if args['mode'] == 'Run':
        OBSERVE = 999999999	#We keep observe, never train
        epsilon = FINAL_EPSILON # higher epsilon, more timestamps?
        print ("Now we load weight")
        model.load_weights("model_v1.h5")
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = adam)
        print ("Weight load successfully")
        t = 0

    elif args['mode'] == 'CTrain': #Continue previous train
        OBSERVE = OBSERVATION
        epsilon = 0.07823368810419994 #0.08811709480229288
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr = LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = adam)
        print ("Weight load successfully")
        t = 0 #360045

    else:					   #We go to training mode -> -m "Train"
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON #o EPSILON é o que divide a parte de exploration vs exploitation. se for abaixo de um dado valor é exploration. Caso contrário é exploitation
        t = 0

    while (True):
        loss = 0
        Q_sa = 0 # Q(s, a) representing the maximum discounted future reward when we perform action a in state s.
        action_index = 0
        r_t = 0 #reward
        a_t = np.zeros([ACTIONS]) #action
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
                
            else:
                q = model.predict(s_t)	   #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduce the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward. DEPOIS DE UM "STEP" correr sempre o "RENDER"
        x_t1_colored, r_t, terminal, info = env.step(np.where(a_t == 1)[0][0]) #FUNÇÂO "WHERE" para obter o índice do valor do array que está a 1
        env.render()
        
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range = (0, 255))


        x_t1 = x_t1 / 255.0


        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE: #train ou update da nossa rede. de quantas em quantas frames vamos precisar para fazer um treino. se replay_mem começar a ficar mto cheio retira a última entrada. e fazemos append das novas decisoes que foram sendo adquiridas
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis = 1) * np.invert(terminal) #qual o target associado

            loss += model.train_on_batch(state_t, targets) #quanto mais proximo de zero, mais proximo está de convergir para conseguir estimar o key value de acordo com o par (estado, ação)

        s_t = s_t1
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite = True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description = 'Description of your program')
    parser.add_argument('-m','--mode', help = 'Train / CTrain / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized\n
            print(e)

    main()