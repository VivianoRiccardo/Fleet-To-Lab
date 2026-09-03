#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
cimport Pyezsparkc
from libc.stdlib cimport free, malloc, srand
from cpython cimport array
from itertools import chain
cimport numpy as npc
import numpy as np
from libc cimport stdint
from libc.stdio cimport printf
from libc.time cimport time
from sys import exit
import random
import math
ctypedef stdint.uint64_t uint64_t
ctypedef stdint.uint8_t uint8_t



PY_ALL_ACTIVATIONS = Pyezsparkc.ALL_ACTIVATIONS
PY_NO_ACTIVATION = Pyezsparkc.NO_ACTIVATION
PY_SIGMOID = Pyezsparkc.SIGMOID
PY_TANH = Pyezsparkc.TANH
PY_RELU = Pyezsparkc.RELU
PY_ELU = Pyezsparkc.ELU
PY_LEAKY_RELU = Pyezsparkc.LEAKY_RELU
PY_GELU = Pyezsparkc.GELU
PY_SOFTMAX = Pyezsparkc.SOFTMAX
PY_N_ACTIVATION_FUNCTIONS_1_0 = Pyezsparkc.N_ACTIVATION_FUNCTIONS_1_0
PY_MAX_POOLING = Pyezsparkc.MAX_POOLING
PY_AVERAGE_POOLING = Pyezsparkc.AVERAGE_POOLING
PY_STORING = Pyezsparkc.STORING
PY_USED_PARAMS = Pyezsparkc.USED_PARAMS
PY_ALL_PARAMS = Pyezsparkc.ALL_PARAMS
PY_PHEROMONE_SCORES = Pyezsparkc.PHEROMONE_SCORES
PY_HEURISTICS = Pyezsparkc.HEURISTICS
PY_PI = Pyezsparkc.PI
PY_EPSILON = Pyezsparkc.EPSILON
PY_SUM_FLAG = Pyezsparkc.SUM_FLAG
PY_MUL_FLAG = Pyezsparkc.MUL_FLAG
PY_DIV_FLAG = Pyezsparkc.DIV_FLAG
PY_DIF_FLAG = Pyezsparkc.DIF_FLAG
PY_EXP_FLAG = Pyezsparkc.EXP_FLAG



cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)



def check_size(vector, dim):
    if type(vector) == np.ndarray:
        shape = vector.shape
        n = 1
        for i in shape:
            n*=i
        if n != dim:
            print("Error: the dimension of one of your arrays is not correct!")
            exit(1)
    elif type(vector) == list:
        v = vector
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        if len(v) != dim:
            print("Error: the dimension of one of your arrays is not correct!")
            exit(1)
    else:
        print("Error: one of your vectors is not a list, neither a np.array!")
        exit(1)
        
def get_size(vector):
    if type(vector) == np.ndarray:
        shape = vector.shape
        n = 1
        for i in shape:
            n*=i
        return n
    elif type(vector) == list:
        v = vector
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        return len(v)
    else:
        print("Error: one of your vectors is not a list, neither a np.array!")
        exit(1)

#if is not multi dimensional there is not good
def get_first_dimension_size(vector):
    v = vector
    if type(v) == list:
        v = np.array(v, dtype=np.float32)
    
    if type(v) == np.ndarray:
        if v.ndim != 2:
            print("your array must be multi dimensional! d 2")
            exit(1)
        else:
            return v.shape[0]
    else:
        print("the input is not a np.array!")
        exit(1)

def get_randomness():
    srand(time(NULL))

def get_randomness_with_seed(seed):
    srand(seed)
    
def get_random_0_1():
    return Pyezsparkc.normalized_random()

    
cdef from_float_to_ndarray(float* ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return np.array(lst, dtype=np.float32)
    
cdef from_int_to_ndarray(int* ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return np.array(lst, dtype=np.int32)
'''    
cdef from_float_to_list(float *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst
'''
#flats every vector (numpy array / list) and create a numpy contiguos array to be used
def vector_is_valid(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return np.ascontiguousarray(np.array(v,dtype=np.float32),dtype=np.float32)
    elif type(v) == np.ndarray:
        if v.ndim > 1:
            v = v.flatten()
        return np.ascontiguousarray(np.array(v,dtype=np.float32),dtype=np.float32) 
    
    print("Error: you vector is not a list neither a numpy.ndarray type")
    exit(1)

def vector_is_valid_int(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return np.ascontiguousarray(np.array(v,dtype=np.intc),dtype=np.intc)
    elif type(v) == np.ndarray:
        if v.ndim > 1:
            v = v.flatten()
        return np.ascontiguousarray(np.array(v,dtype=np.intc),dtype=np.intc) 
    
    print("Error: you vector is not a list neither a numpy.ndarray type")
    exit(1)

def vector_is_valid_uint64(vector):
    v = vector
    if type(v) == list:
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*v))
        return np.ascontiguousarray(np.array(v,dtype=np.uint64),dtype=np.uint64)
    elif type(v) == np.ndarray:
        if v.ndim > 1:
            v = v.flatten()
        return np.ascontiguousarray(np.array(v,dtype=np.uint64),dtype=np.uint64) 
    
    print("Error: you vector is not a list neither a numpy.ndarray type")
    exit(1)

def check_int(int i):
    if i < 0 or i >= 2**31-1:
        print("Error, wrong input to pass")
        exit(1)
        
def check_float(float i):
    if i >= 2**31-1:
        print("Error, wrong input to pass")
        exit(1)

def is_square_matrix(matrix):
    if not isinstance(matrix, list):
        return False
    
    if not all(isinstance(row, list) for row in matrix):
        return False
    
    n = len(matrix)
    
    if n == 0:
        return False
    
    if not all(len(row) == n for row in matrix):
        return False
    
    return True


        

cdef class acoActivation:
    
    cdef Pyezsparkc.aco_activation* _activation
    cdef Pyezsparkc.aco_node* _node
    cdef int activation_flag
    cdef int activations_size
    cdef int input_size
    cdef int output_size
    cdef int identifier
    
    
    
    def __cinit__(self,int activation_flag,int activations_size,int input_size,int output_size, int identifier):
        if activation_flag != PY_ALL_ACTIVATIONS and activation_flag != PY_NO_ACTIVATION and activation_flag != PY_SIGMOID and activation_flag != PY_TANH and activation_flag != PY_RELU and activation_flag != PY_ELU and activation_flag != PY_LEAKY_RELU and activation_flag != PY_GELU  and activation_flag != PY_SOFTMAX:
            print("Error: activation no recognized!")
            exit(1)
        if activations_size != 0 and activations_size != PY_N_ACTIVATION_FUNCTIONS_1_0:
            print("Error: activation size can only be either 0 or PY_N_ACTIVATION_FUNCTIONS_1_0")
            exit(1)
        check_int(input_size)
        check_int(output_size)
        check_int(identifier)
        self.activation_flag = activation_flag
        self.activations_size = activations_size
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
        
    
    def get_mode(self):
        cdef int* activations = NULL;
        if self.activations_size == PY_ALL_ACTIVATIONS:
            activations = Pyezsparkc.generate_all_activations()
            self._activation = Pyezsparkc.init_aco_activation(self.activation_flag, self.activations_size, self.input_size, self.output_size,self.identifier, 0,0,0,NULL,NULL,NULL, &activations[0])
        self._activation = Pyezsparkc.init_aco_activation(self.activation_flag, self.activations_size, self.input_size, self.output_size,self.identifier, 0,0,0,NULL,NULL,NULL, NULL)
        
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,NULL,NULL,NULL,self._activation,NULL,NULL,NULL,NULL);
        
        

cdef class acoCl:
    
    cdef int padding_rows
    cdef int padding_cols
    cdef int input_rows
    cdef int input_cols
    cdef int channels
    cdef int kernel_rows
    cdef int kernel_cols
    cdef int n_kernels
    cdef int identifier
    cdef int stride_rows
    cdef int stride_cols
    cdef Pyezsparkc.aco_cl* _cl
    cdef Pyezsparkc.aco_node* _node
    

    
    def __cinit__(self,int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, int identifier, int stride_rows, int stride_cols):
        check_int(padding_rows)
        check_int(padding_cols)
        check_int(input_rows)
        check_int(input_cols)
        check_int(channels)
        check_int(kernel_rows)
        check_int(kernel_cols)
        check_int(n_kernels)
        check_int(identifier)
        check_int(stride_rows)
        check_int(stride_cols)
        
        self.padding_rows = padding_rows
        self.padding_cols = padding_cols
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.channels = channels
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.n_kernels = n_kernels
        self.identifier = identifier
        self.stride_rows = stride_rows
        self.stride_cols = stride_cols
    def get_mode(self):
        cdef Pyezsparkc.aco_cl* _convolution
        self._cl = Pyezsparkc.init_aco_cl(self.padding_rows, self.padding_cols, self.input_rows, self.input_cols, self.channels, self.kernel_rows, self.kernel_cols, self.n_kernels, self.identifier, self.stride_rows, self.stride_cols, 0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,self._cl,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

cdef class acoConcatenation:
    
    cdef int input_size
    cdef int output_size
    cdef float value
    cdef int identifier
    cdef Pyezsparkc.aco_concatenation* _concatenation
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self,int input_size, int output_size, float value, int identifier):
        check_int(input_size)
        check_int(output_size)
        check_int(identifier)
        check_float(value)
        
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
        self.value = value
    def get_mode(self):
        cdef Pyezsparkc.aco_cl* _concatenation
        self._concatenation = Pyezsparkc.init_aco_concatenation(self.input_size, self.output_size,self.value,self.identifier, 0,0,0,NULL,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,NULL,NULL,self._concatenation,NULL,NULL,NULL,NULL,NULL);

cdef class acoFcl:
    
    cdef int input_size
    cdef int output_size
    cdef int identifier
    cdef Pyezsparkc.aco_fcl* _fcl
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self,int input_size, int output_size,  int identifier):
        check_int(input_size)
        check_int(output_size)
        check_int(identifier)
        
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
    def get_mode(self):
        cdef Pyezsparkc.aco_cl* _fully_connected
        self._fcl = Pyezsparkc.init_aco_fcl(self.input_size,self.output_size,self.identifier,0, 0,0, 0, 0,0, NULL, NULL, NULL, NULL, NULL, NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,self._fcl,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);

cdef class acoFclKolmogorovArnold:
    
    cdef int input_size
    cdef int output_size
    cdef int identifier
    cdef int n_functions
    cdef int maximum_degree
    cdef Pyezsparkc.aco_fcl_kolmogorov_arnold* _fcl_kolmogorov_arnold
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self, int input_size, int output_size,int n_functions, int maximum_degree, int identifier):
        check_int(input_size)
        check_int(output_size)
        check_int(n_functions)
        check_int(maximum_degree)
        check_int(identifier)
        
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
        self.maximum_degree = maximum_degree
        self.n_functions = n_functions
    def get_mode(self):
        cdef Pyezsparkc.aco_fcl_kolmogorov_arnold* _fcl_kolmogorov_arnold
        self._fcl_kolmogorov_arnold = Pyezsparkc.init_aco_fcl_kolmogorov_arnold(self.input_size, self.output_size,self.n_functions, self.maximum_degree, self.identifier, 0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,NULL,NULL,NULL,NULL,self._fcl_kolmogorov_arnold,NULL,NULL,NULL);
        
cdef class acoLstm:
    
    cdef int input_size
    cdef int output_size
    cdef int identifier
    cdef int timestamp
    cdef int window
    cdef Pyezsparkc.aco_lstm* _lstm
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self, int timestamp, int input_size, int output_size,int window, int identifier):
        check_int(input_size)
        check_int(output_size)
        check_int(window)
        check_int(timestamp)
        check_int(identifier)
        
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
        self.timestamp = timestamp
        self.window = window
    def get_mode(self):
        cdef Pyezsparkc.aco_lstm* _lstm
        self._lstm = Pyezsparkc.init_aco_lstm(self.timestamp, self.input_size, self.output_size, self.window, self.identifier, 0,0,0,0,0,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,self._lstm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
        
cdef class acoPooling:
    
    cdef int padding_rows
    cdef int padding_cols
    cdef int input_rows
    cdef int input_cols
    cdef int channels
    cdef int pooling_rows
    cdef int pooling_cols
    cdef int stride_rows
    cdef int stride_cols
    cdef int identifier
    cdef int pooling_type
    cdef Pyezsparkc.aco_pooling* _pooling
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self, int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int identifier, int pooling_type):
        check_int(padding_rows)
        check_int(padding_cols)
        check_int(input_rows)
        check_int(input_cols)
        check_int(channels)
        check_int(pooling_rows)
        check_int(pooling_cols)
        check_int(stride_rows)
        check_int(stride_cols)
        check_int(identifier)
        check_int(pooling_type)
        
        if pooling_type != PY_MAX_POOLING and pooling_type != PY_AVERAGE_POOLING:
            print("Error: pooling type can onl be either PY_MAX_POOLING or PY_AVERAGE_POOLING")
            exit(1)
        
        self.padding_rows = padding_rows
        self.padding_cols = padding_cols
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.channels = channels
        self.pooling_rows = pooling_rows
        self.padding_rows = padding_rows
        self.pooling_cols = pooling_cols
        self.stride_rows = stride_rows
        self.stride_cols = stride_cols
        self.identifier = identifier
        self.pooling_type = pooling_type
    def get_mode(self):
        cdef Pyezsparkc.aco_pooling* _pooling
        self._pooling = Pyezsparkc.init_aco_pooling(self.padding_rows, self.padding_cols, self.input_rows, self.input_cols, self.channels, self.pooling_rows, self.pooling_cols, self.stride_rows, self.stride_cols, self.identifier, self.pooling_type,0,0,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,NULL,self._pooling,NULL,NULL,NULL,NULL,NULL,NULL);
        
        
cdef class acoSum:
    
    cdef int input_size
    cdef int input2_size
    cdef int output_size
    cdef int identifier
    cdef int flag
    cdef Pyezsparkc.aco_sum* _sum
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self,int input_size, int input2_size, int output_size,int identifier, int flag = PY_SUM_FLAG):
        if flag != PY_SUM_FLAG and flag != PY_MUL_FLAG and flag != PY_DIV_FLAG and flag != PY_DIF_FLAG and flag != PY_EXP_FLAG:
            print("Error: aco sum flag can only beeither PY_SUM_FLAG or PY_DIV_FLAG or PY_DIF_FLAG or PY_EXP_FLAG or PY_MUL_FLAG")
            exit(1)
        check_int(input_size)
        check_int(input2_size)
        check_int(output_size)
        check_int(identifier)
        
        self.input_size = input_size
        self.input2_size = input2_size
        self.output_size = output_size
        self.identifier = identifier
        self.flag = flag
    def get_mode(self):
        cdef Pyezsparkc.aco_sum* _sum
        self._sum= Pyezsparkc.init_aco_sum(self.input_size, self.input2_size, self.output_size,self.identifier, self.flag,0,0,0,NULL,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,self._sum,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
        
cdef class acoSplitter:
    
    cdef int input_size
    cdef int output_size
    cdef int identifier
    cdef Pyezsparkc.aco_splitter* _splitter
    cdef Pyezsparkc.aco_node* _node
    
    
    def __cinit__(self,int input_size, int output_size,int identifier):
        check_int(input_size)
        check_int(output_size)
        check_int(identifier)
        
        self.input_size = input_size
        self.output_size = output_size
        self.identifier = identifier
    def get_mode(self):
        cdef Pyezsparkc.aco_splitter* _splitter
        self._splitter= Pyezsparkc.init_aco_splitter(self.input_size, self.output_size,self.identifier, 0,0,0,NULL,NULL)
        
    def get_node(self):
        self.get_mode()
        self._node = Pyezsparkc.init_aco_node(0,0,self.identifier,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,self._splitter,NULL,NULL);
        
        



cdef class acoSuperstruct:
    cdef Pyezsparkc.aco_superstruct* _model
    cdef Pyezsparkc.aco_superstruct** _models
    cdef float* _inputs
    cdef int number_of_inputs
    cdef int input_size
    cdef int threads
    cdef int number_of_parameters
    cdef int is_copied
    cdef int explorative_heuristic
    cdef int only_for_inference
    
    
    def __cinit__(self,list activations, list convolutions, list concatenations, list fullyconnected, list kolmogorovfullyconnected, list lstms, list poolings, list sums, list adjacience_matrix, list same_parameters):
        n_nodes = 0
        counter = 0
        n_nodes+=len(activations)
        n_nodes+=len(concatenations)
        n_nodes+=len(fullyconnected)
        n_nodes+=len(convolutions)
        n_nodes+=len(kolmogorovfullyconnected)
        n_nodes+=len(lstms)
        n_nodes+=len(poolings)
        n_nodes+=len(sums)
        cdef acoActivation a
        cdef acoCl b
        cdef acoConcatenation c
        cdef acoFcl d
        cdef acoFclKolmogorovArnold e
        cdef acoLstm f
        cdef acoPooling g
        cdef acoSum h
        
        if not is_square_matrix(adjacience_matrix) or n_nodes != len(adjacience_matrix):
            print("Error: the matrix must be nXn where n is the number of nodes");
            exit(1);
        if not is_square_matrix(same_parameters) or n_nodes != len(same_parameters):
            print("Error: the same_parameters matrix must be nXn where n is the number of nodes");
            exit(1);
        
        cdef Pyezsparkc.aco_node** _aco_nodes
        _aco_nodes = <Pyezsparkc.aco_node**>malloc(n_nodes*sizeof(Pyezsparkc.aco_node*))
        for i in range(len(activations)):
            a = activations[i]
            a.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>a._node
            counter+=1
        for i in range(len(concatenations)):
            c = concatenations[i]
            c.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>c._node
            counter+=1
        for i in range(len(fullyconnected)):
            d = fullyconnected[i]
            d.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>d._node
            counter+=1
        for i in range(len(convolutions)):
            b = convolutions[i]
            b.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>b._node
            counter+=1
        for i in range(len(kolmogorovfullyconnected)):
            e = kolmogorovfullyconnected[i]
            e.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>e._node
            counter+=1
        for i in range(len(lstms)):
            f = lstms[i]
            f.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>f._node
            counter+=1
        for i in range(len(poolings)):
            g = poolings[i]
            g.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>g._node
            counter+=1
        for i in range(len(sums)):
            h = sums[i]
            h.get_node()
            _aco_nodes[counter] = <Pyezsparkc.aco_node*>h._node
            counter+=1
        
        cdef int** c_matrix = <int**>malloc(sizeof(int*)*n_nodes)
        cdef int** c_same_params = <int**>malloc(sizeof(int*)*n_nodes)
        for i in range(n_nodes):
            c_matrix[i] = <int*>malloc(sizeof(int)*n_nodes)
            c_same_params[i] = <int*>malloc(sizeof(int)*n_nodes)
            for j in range(n_nodes):
                c_matrix[i][j] = adjacience_matrix[i][j]
                c_same_params[i][j] = same_parameters[i][j]
        
        self._model = Pyezsparkc.init_aco_superstruct(_aco_nodes,c_matrix,c_same_params,n_nodes)
        Pyezsparkc.generate_parameters_array_to_superstruct(self._model)
        self.number_of_parameters = 1
        self.threads = 1
        self.number_of_inputs = 0
        self.input_size = Pyezsparkc.aco_superstruct_get_total_input_size(self._model)
        self._inputs = NULL
        self.is_copied = 0
        self.explorative_heuristic = 0
        self.only_for_inference = -1
        
        
    def __dealloc__(self):
        if self.threads > 1:
            for i in range(1,self.threads):
                Pyezsparkc.free_aco_superstruct(self._models[i])
            free(self._models)
        if self.is_copied == 1:
            Pyezsparkc.aco_superstruct_set_all_params_to_null(self._model)
        Pyezsparkc.free_aco_superstruct(self._model)
        
    def set_only_for_inference(self):
        if self.only_for_inference == -1:
            Pyezsparkc.set_aco_superstruct_only_for_inference(self._model)
            self.only_for_inference = 1

    
    def copy_pheromone(self,acoSuperstruct s):
        if self._model is NULL:
            return
        Pyezsparkc.copy_aco_superstruct_array(<Pyezsparkc.aco_superstruct*> s._model, self._model, 3)

        
    def initialize_pso(self, number_of_parameters):
        if self._model is NULL:
            return
        check_int(number_of_parameters)
        self.number_of_parameters = number_of_parameters
        Pyezsparkc.generate_pso_parameters_array_to_superstruct(self._model, number_of_parameters)
        Pyezsparkc.generate_aco_pheromone_parameters_array_to_superstruct(self._model, 0.0)
        Pyezsparkc.generate_aco_pso_y_alphas_array_to_superstruct(self._model, 0.0)
    
    def py_set_pheromone_for_index(self,  int index, float pheromone):
        if self._model is NULL:
            return
        check_int(index)
        check_float(pheromone)
        Pyezsparkc.set_pheromone_for_index(self._model,index,pheromone)


    
    def initialize_aco_eiwoa(self, number_of_parameters = 100, initial_pheromone = 10,number_of_ants = 100, number_of_single_aco_iteration = 26, number_of_total_iterations = 250, number_of_x_iterations = 10, tau_min = 0, tau_max = 10, p_dec = 0.95, p = 0.5, rho = 0.5, time_to_update_with_global_best = 8, k_zero = 0.7, alpha_aco = 1.0, beta_aco = 1.0, max_alpha_aco = 1.0, max_beta_aco = 1.0,min_alpha_aco = 1.0, min_beta_aco = 1.0,  exponential_alpha_decay = 1.0, exponential_beta_decay = 1.0):
        if self._model is NULL:
            return
        check_int(number_of_parameters)
        check_float(initial_pheromone)
        check_int(number_of_single_aco_iteration)
        check_int(number_of_total_iterations)
        check_int(number_of_x_iterations)
        check_int(time_to_update_with_global_best)
        check_float(tau_min)
        check_float(tau_max)
        check_float(p_dec)
        check_float(p)
        check_float(rho)
        check_float(k_zero)
        check_float(alpha_aco)
        check_float(beta_aco)
        check_float(max_alpha_aco)
        check_float(max_beta_aco)
        check_float(min_alpha_aco)
        check_float(min_beta_aco)
        check_float(exponential_alpha_decay)
        check_float(exponential_beta_decay)
        
        if number_of_parameters <= 0:
            print("Error, the number of ants can't be <= 0!")
            exit(1)
        if initial_pheromone <0:
            print("Error, the initial pheromone can't be < 0!")
            exit(1)
        if number_of_single_aco_iteration <= 0:
            print("Error, the number_of_single_aco_iteration can't be <= 0!")
            exit(1)
        if number_of_total_iterations < number_of_single_aco_iteration:
            print("Error, the number of total iterations can't be < number of single aco iteration!")
            exit(1)
        if number_of_total_iterations/number_of_single_aco_iteration > number_of_x_iterations:
            print("Error, can't be number_of_total_iterations/number_of_single_aco_iteration > number_of_x_iterations")
            exit(1)
        if tau_min < 0:
            print("Error, tau_min can't be < 0!")
            exit(1)
        if tau_max < tau_min:
            print("Error, tau_max can't be < tau_min")
            exit(1)
        if p_dec < 0 or p_dec > 1:
            print("Error, p_dec must be in [0,1]")
            exit(1)
        if p < 0 or p > 1:
            print("Error, p must be in [0,1]")
            exit(1)
        if rho < 0 or rho > 1:
            print("Error, rho must be in [0,1]")
            exit(1)
        if k_zero < 0 or k_zero > 1:
            print("Error, k_zero must be in [0,1]")
            exit(1)
        if time_to_update_with_global_best < 0:
            print("Error, time_to_update_with_global_best can't be < 0!")
            exit(1)
        if self.only_for_inference != -1:
            print("Error, you have already given a setup for inference or training for this model!")
            exit(1)
         
        
        
        
        Pyezsparkc.generate_aco_parameters_array_to_superstruct(self._model, number_of_parameters)
        Pyezsparkc.generate_aco_pheromone_parameters_array_to_superstruct(self._model, initial_pheromone)
        Pyezsparkc.aco_superstruct_activate_heuristic(self._model,100,0.99,alpha_aco,beta_aco,1.0);
        Pyezsparkc.generate_aco_superstruct_bit_arrays(self._model)
        Pyezsparkc.init_aco_superstruct_general_stuff(self._model,number_of_single_aco_iteration,number_of_total_iterations,number_of_x_iterations,number_of_ants,tau_min,tau_max,p_dec,p,rho,time_to_update_with_global_best,k_zero)
        Pyezsparkc.init_aco_superstruct_eiwoa_stuff(self._model,1.0,2.0,1.0,1.0,2.0, 5,-5)
        Pyezsparkc.set_alpha_and_beta_aco(self._model, alpha_aco,beta_aco)
        Pyezsparkc.set_max_aco_alpha_beta(self._model, max_alpha_aco,max_beta_aco, min_alpha_aco, min_beta_aco, exponential_alpha_decay,exponential_beta_decay)
        self.only_for_inference = 0
    
    def update_alpha_beta_params(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.update_alpha_beta_params(self._model)
    
    def set_explorative_heuristic(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.aco_superstruct_activate_explorative_supporting_heuristic_array(self._model)
        self.explorative_heuristic = 1
        
    def save_current_weights(self, int n):
        if self._model is NULL:
            return
        check_int(n)
        if n < 0:
            print("Error n can't be < 0")
        Pyezsparkc.save_weights_single_model(self._model, n)
        
    def load_current_weights(self, weights_filename):
        if self._model is NULL:
            return
        cdef char* ss1
        ss1 = <char*>PyUnicode_AsUTF8(weights_filename)
        Pyezsparkc.load_weights_single_model(self._model, ss1)

    
    def save_weights_and_pheromones(self, int n):
        if self._model is NULL or self.only_for_inference != 0:
            return
        check_int(n)
        if n < 0:
            print("Error n can't be < 0")
        Pyezsparkc.save_weights_and_pheromone_aco_superstruct(self._model, n)
    
    def load_weights_and_pheromones(self, weights_filename, pheromone_filename):
        if self._model is NULL or self.only_for_inference != 0:
            return
        cdef char* ss1
        cdef char* ss2
        ss1 = <char*>PyUnicode_AsUTF8(pheromone_filename)
        ss2 = <char*>PyUnicode_AsUTF8(weights_filename)
        Pyezsparkc.load_weights_and_pheromone_aco_superstruct(self._model,  ss1,ss2)

    
    def set_alpha_and_beta_aco(self, alpha_aco, beta_aco):
        if self._model is NULL or self.only_for_inference != 0:
            return
        check_float(alpha_aco)
        check_float(beta_aco)
        Pyezsparkc.set_alpha_and_beta_aco(self._model, alpha_aco,beta_aco)
        
    def random_select_network(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight(self._model)
        if self.threads > 1:
            for z in range(self.threads):
                Pyezsparkc.copy_aco_superstruct_activation(self._models[z],self._model)
    def select_best_network(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.aco_superstruct_select_params_according_to_best_pheromone(self._model)
        if self.threads > 1:
            for z in range(self.threads):
                Pyezsparkc.copy_aco_superstruct_activation(self._models[z],self._model)
            

        
        
    def reset_pheromone_according_to_index(self, index):
        if self._model is NULL or self.only_for_inference != 0:
            return
        if index != 0 and index != 1 and index != 2:
            print("Error, index can be only 0 (global best, 1, current best, 2, current path)")
            exit(1)
        Pyezsparkc.reset_pheromone_according_to_index(self._model, index)

    def set_local_pheromone(self, pheromone):
        if self._model is NULL or self.only_for_inference != 0:
            return
        check_float(pheromone)
        if pheromone < 0:
            print("Error, pheromone can't be negative!")
            exit(1)
        Pyezsparkc.set_local_pheromone(self._model, pheromone)

    def replace_local_best_with_current(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        
        Pyezsparkc.check_local_best_pheromone(self._model)


    def replace_global_best_with_best_local(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        if self.explorative_heuristic == 0:
            Pyezsparkc.check_global_best_pheromone(self._model)
        else:
            Pyezsparkc.check_global_best_pheromone_without_updating_heuristic(self._model)
    def update_taus(self):
        if self._model is NULL or self.only_for_inference != 0:
            return 
        Pyezsparkc.aco_superstruct_update_taus(self._model)
    
    def update_pheromone_according_to_local_best(self):
        if self._model is NULL or self.only_for_inference != 0:
            return 
        Pyezsparkc.update_pheromone_according_to_local_best(self._model)

    
    def update_current_p(self):
         if self._model is NULL or self.only_for_inference != 0:
             return
         Pyezsparkc.update_current_p(self._model)
    
    def update_pheromone_according_to_global_best(self):
        if self._model is NULL:
            return
        Pyezsparkc.update_pheromone_according_to_global_best(self._model)
    
    def update_heuristic_according_to_global_best(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.update_heuristics(self._model)
    
    def increase_explorative_heuristic(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.aco_increase_explorative_supporting_heuristic_array(self._model)
    
    def update_heuristic_according_to_explorative_one(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.update_heuristics_according_to_explorative_supporting_heuristic_array(self._model)
    
    def update_eiwoa(self):
        if self._model is NULL:
            return
        Pyezsparkc.aco_superstruct_update_params_eiwoa(self._model)
        Pyezsparkc.update_eiwoa_params(self._model)
    
    def reset_explorative_heuristic(self):
        if self._model is NULL or self.only_for_inference != 0:
            return
        Pyezsparkc.reset_explorative_heuristics_support(self._model)
        Pyezsparkc.reset_heuristic_all_ones(self._model)
        
    def make_multi_thread(self, int threads):
        if self._model is NULL:
            return
        check_int(threads)
        if threads <= 1:
            return
        self.threads = threads
        self._models = <Pyezsparkc.aco_superstruct**>malloc(threads*sizeof(Pyezsparkc.aco_superstruct*))
        self._models[0] = self._model
        for i in range(1,threads):
            self._models[i] = Pyezsparkc.copy_aco_superstruct_with_only_storing_params(self._model)
    def select_params_according_to_index_pso(self, int index):
        if self._model is NULL or self.only_for_inference != 0:
            return
        check_int(index)
        if index >= self.number_of_parameters:
            print("Error: Your index is out of range")
            exit(1)
        Pyezsparkc.aco_superstruct_select_params_according_to_index_pso(self._model,index)
        
    def check_input_size(self, n):
        if n < self.input_size or n%self.input_size > 0:
            print("Error: the size of your inputs is incorrect, it must be a multiple of "+str(self.input_size)+", your size is "+str(n))
            exit(1)
            
    def forward(self, inputs):
        if self._model is NULL:
            print("Error: you did'nt setup any model")
            exit(1)
        if self.number_of_parameters < 1:
            print("Error: you didn't setup any input")
            exit(1)
            
        cdef int n = get_size(inputs)
        if n < self.input_size or n%self.input_size > 0:
            print("Error: the size of your inputs is incorrect, it must be a multiple of "+str(self.input_size)+", your size is "+str(n))
            exit(1)
        self.number_of_inputs = <int>(n/self.input_size)
        cdef float[:] v = vector_is_valid(inputs)
        cdef float* network_output = NULL

        if self.threads == 1:
            network_output = Pyezsparkc.get_multi_output_from_multi_input_aco_superstruct_ff(&self._model,<float*>&v[0],self.threads,self.number_of_inputs)
        else:
            network_output = Pyezsparkc.get_multi_output_from_multi_input_aco_superstruct_ff(self._models,<float*>&v[0],self.threads,self.number_of_inputs)
        ret = from_float_to_ndarray(network_output,self.number_of_inputs*Pyezsparkc.aco_superstruct_get_total_output_size(self._model))
        free(network_output)
        return ret
    
    def get_total_output_size_with_n(self, n):
        return (<int>(n/self.input_size))*Pyezsparkc.aco_superstruct_get_total_output_size(self._model)
    
    def get_total_output_size_true_value(self):
        return Pyezsparkc.aco_superstruct_get_total_output_size(self._model)
    
    def set_loss(self, int index, double loss):
        check_int(index)
        if index < 0 or index >= self.number_of_parameters:
            print("Error: index out of range")
            exit(1)
        if self._model is NULL:
            print("Error: you did'nt setup any model")
            exit(1)
        Pyezsparkc.set_losses_according_to_index(self._model,index,loss)

    def increase_internal_iteration_index(self):
        if self._model is NULL:
            return
        Pyezsparkc.increase_iteration_index(self._model)

    
    def copy_array_weights_to_specific_struct_weight_position(self, array, int index, int size):
        cdef float[:] v = vector_is_valid(array)
        Pyezsparkc.copy_array_weights_to_specific_struct_weight_position(self._model, <float*>&v[0], index, size)
    
    def get_current_weight_combination_to_output_array(self, size):
        cdef float* network_output = NULL
        network_output = Pyezsparkc.get_current_weight_combination_to_output_array(self._model)
        ret = from_float_to_ndarray(network_output, size)
        free(network_output)
        return ret
        
    def get_current_indices_local_best(self):
        cdef int* network_output = NULL
        cdef int size = Pyezsparkc.get_indices_size(self._model)
        network_output = Pyezsparkc.get_indices_from_local_best(self._model)
        ret = from_int_to_ndarray(network_output, size)
        free(network_output)
        return ret

