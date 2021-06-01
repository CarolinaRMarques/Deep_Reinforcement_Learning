import os

print("Please provide the necessary data:")

model = input('Model to execute. \n')
mode = input('Mode of the program: Run , Train, CTrain.\n')
episodes = input('Number of episodes.\n')
batch_size = input('Size of the batch to train with.\n')
learning_rate = input('LearningRate \n')
initial_epsilon = input('Rate of exploration.\n')
final_epsilon =input('Rate of exploration.\n')
max_memory = input('Memory Replay maximum size. \n')
D = input('Discount for the epsilon.\n')
explore = input('Frames over which to anneal epsilon \n')
Observation = input('Timesteps to observe before training \n')



## Invocar o modelo com os parametros definidos
os.system("python3 " + model + " " +  mode + " " +  episodes + " " + batch_size + " " + learning_rate + " " + initial_epsilon + " " + final_epsilon + " " +  max_memory + " " + D + " " +  explore + " " +  Observation ) 