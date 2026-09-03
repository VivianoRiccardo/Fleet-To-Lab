import pyezsparkc
import numpy as np

def load_data():
    f = open('./data/train.bin','rb')
    s = f.read().decode('utf-8')
    f.close()
    l_in = []
    l_out = []
    for i in range(50000):
        inputs = []
        outputs = [0,0,0,0,0,0,0,0,0,0]
        for j in range(0,28*28+1):
            if j < 28*28:
                inputs.append(int(s[i*(28*28+1)+j:i*(28*28+1)+j+1]))
            else:
                outputs[int(s[i*(28*28+1)+j:i*(28*28+1)+j+1])] = 1
        l_in.append(inputs)
        l_out.append(outputs)
    return np.array(l_in, dtype='float'), np.array(l_out,dtype="float")

def shuffle_the_arrays(A,B):
    permutation = np.random.permutation(50000)
    A_shuffled = A[permutation]
    B_shuffled = B[permutation]
    return A_shuffled, B_shuffled
    
def cross_entropy(true_distribution, predicted_distribution):
    epsilon = 1e-15
    predicted_distribution = np.clip(predicted_distribution, epsilon, 1 - epsilon)
    cross_entropy_loss = -np.sum(true_distribution * np.log(predicted_distribution))
    return cross_entropy_loss
    
number_of_parameters = 100
batch_size = 200
epochs = 250
threads = 16
initial_pheromone = 10

fcl1 = pyezsparkc.acoFcl(28*28, 100, 0)
fcl2 = pyezsparkc.acoFcl(100, 100, 2)
fcl3 = pyezsparkc.acoFcl(100, 10, 4)
activation1 = pyezsparkc.acoActivation(pyezsparkc.PY_RELU,0,100,100,1)
activation2 = pyezsparkc.acoActivation(pyezsparkc.PY_RELU,0,100,100,3)
activation3 = pyezsparkc.acoActivation(pyezsparkc.PY_SOFTMAX,0,10,10,5)
number_of_nodes = 6
matrix = []
params = []

for i in range(number_of_nodes-1):
    l = [0]*number_of_nodes
    l[i+1] = 1
    matrix.append(l) 
    params.append([0]*number_of_nodes)

params.append([0]*number_of_nodes)
matrix.append([0]*number_of_nodes)

model = pyezsparkc.acoSuperstruct([activation1,activation2,activation3],[],[],[fcl1,fcl2,fcl3],[],[],[],[],matrix,params)
model.initialize_aco_eiwoa(number_of_parameters, initial_pheromone)
model.make_multi_thread(threads)
inputs, outputs = load_data()
    
min_loss = 99999999999
count_update_x = 0
time_to_update_with_global_best = 8
iteration_index = 0
global_best_before = -1
number_iterations_for_single_aco_iteration = 26

print("training")
for i in range(epochs):
    model.reset_pheromone_according_to_index(1)
    model.reset_pheromone_according_to_index(2)
    global_best = 0
    for k in range(number_of_parameters):
        model.random_select_network()
        network_output = model.forward(inputs)
        loss = cross_entropy(outputs.flatten(),network_output)
        pheromone = 500000.0/loss
        if pheromone > global_best:
            global_best = pheromone
        model.set_local_pheromone(pheromone)
        model.replace_local_best_with_current()
    iteration_index += 1
    count_update_x +=1
    model.increase_internal_iteration_index()
    model.replace_global_best_with_best_local()
    model.update_taus()
    model.update_pheromone_according_to_local_best()
    model.update_current_p()
    
    if count_update_x%time_to_update_with_global_best == 0:
        model.update_pheromone_according_to_global_best()
    
    if global_best > global_best_before:
        model.update_heuristic_according_to_global_best()
        global_best_before = global_best
    if iteration_index%number_iterations_for_single_aco_iteration== 0:
        model.update_eiwoa()
        count_update_x = 0
        model.update_heuristic_according_to_global_best()
    print(global_best)
