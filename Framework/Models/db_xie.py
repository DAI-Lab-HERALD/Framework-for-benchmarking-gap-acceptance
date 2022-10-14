import numpy as np
import pandas as pd
import abc
from model_template import model_template

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    
class RBM(abc.ABC):
    
    def __init__(self, n_visible, n_hidden, learning_rate = 0.01, momentum = 0.95, xavier_const = 1.0):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')
            
        def xavier_init(fan_in, fan_out, *, const=1.0, dtype=tf.dtypes.float32):
            k = const * np.sqrt(6.0 / (fan_in + fan_out))
            return tf.random.uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.w = tf.Variable(xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

    

    def step(self, x):
        def sample_bernoulli(ps):
            return tf.nn.relu(tf.sign(ps - tf.random.uniform(tf.shape(ps))))
        
        x = tf.constant(x.astype('float32'))
        
        hidden_p = tf.nn.sigmoid(tf.matmul(x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        self.delta_w = self._apply_momentum(self.delta_w,
                                            positive_grad - negative_grad)
        self.delta_visible_bias = self._apply_momentum(self.delta_visible_bias,
                                                       tf.reduce_mean(x - visible_recon_p, 0))
        self.delta_hidden_bias = self._apply_momentum(self.delta_hidden_bias,
                                                      tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        self.w.assign_add(self.delta_w)
        self.visible_bias.assign_add(self.delta_visible_bias)
        self.hidden_bias.assign_add(self.delta_hidden_bias)

        return tf.reduce_mean(tf.square(x - visible_recon_p))

    def _apply_momentum(self, old, new):
        n = tf.cast(new.shape[0], dtype=tf.float32)
        m = self.momentum
        lr = self.learning_rate

        return tf.add(tf.math.scalar_mul(m, old), tf.math.scalar_mul((1 - m) * lr / n, new))

    def fit(self, X, epoches = 10, batch_size = 10):
        assert epoches >= 0, "Number of epoches must be positive"
        Index = np.arange(len(X))
        for epoch in range(epoches):
            np.random.shuffle(Index)
            for batch in range(int(np.floor(len(X)/batch_size))):
                Xb = X[Index[batch * batch_size:(batch + 1) * batch_size]]
                self.step(Xb)

    def get_state(self):
        return {'w': self.w,
                'visible_bias': self.visible_bias,
                'hidden_bias': self.hidden_bias,
                'delta_w': self.delta_w,
                'delta_visible_bias': self.delta_visible_bias,
                'delta_hidden_bias': self.delta_hidden_bias
                }




class deep_belief_network():
    def __init__(self, structure):
        self.structure = structure
    
    def train(self, X, Y, RBM_weights_old, batch_size = 10, epochs_pre = 100, epochs = 100):
        if not isinstance(X,np.ndarray):
            raise TypeError("Training data is supposed to be numpy array")
        if X.ndim != 2:
            raise ValueError("Training data is supposed to be a 2D numpy array")
        if X.shape[1] != self.structure[0]:
            raise ValueError("Traing data does not match network input size")
            
        if not isinstance(Y,np.ndarray):
            raise TypeError("Training data is supposed to be numpy array")
        if Y.ndim != 2:
            raise ValueError("Training data is supposed to be a 2D numpy array")
        if Y.shape[1] != self.structure[-1]:
            raise ValueError("Traing data does not match network output size") 
        
        # Pretrain network
        number_RBM = len(self.structure) - 2
        number_RBM_pretrained = len(RBM_weights_old)
        
        assert number_RBM > number_RBM_pretrained or number_RBM == 0, "Error"
        
        W_RBM = []
        B_RBM = []
        X_i = np.copy(X)
        for i in range(number_RBM):
            if i < number_RBM_pretrained:
                W_i, b_i = RBM_weights_old[i]
                
                assert W_i.shape == (self.structure[i], self.structure[i+1])
                assert b_i.size == self.structure[i+1]
            else:
                RBM_i = RBM(self.structure[i],self.structure[i+1])
                RBM_i.fit(X_i, epochs_pre, batch_size)
                data_i = RBM_i.get_state()
                W_i = data_i['w'].numpy()
                b_i = data_i['hidden_bias'].numpy()
            X_i = 1 / (1 + np.exp(np.dot(X_i,W_i)+b_i))
            W_RBM.append(W_i)
            B_RBM.append(b_i)
        
        RBM_weights = list(map(list, zip(* [W_RBM,B_RBM])))
        # Build deep belief network
        self.In = tf.keras.layers.Input(self.structure[0])
        self.dbn = tf.keras.layers.Dense(self.structure[1],activation='sigmoid')(self.In)
        for i in range(2,len(self.structure)-1):
            self.dbn = tf.keras.layers.Dense(self.structure[i],activation='sigmoid')(self.dbn)
        self.dbn = tf.keras.layers.Dense(self.structure[-1],activation='softmax')(self.dbn)
        
        self.DBN = tf.keras.models.Model(self.In,self.dbn)
        
        for i in range(1, 1 + number_RBM):
            self.DBN.layers[i].set_weights([W_RBM[i-1],B_RBM[i-1]])
            self.DBN.layers[i].trainable=False
        
        self.DBN.compile('Adam','binary_crossentropy')
        self.DBN.fit(X, Y, batch_size = batch_size, epochs = epochs, verbose = 0)
        
        self.DBN.trainable=True
        self.DBN.compile('Adam','binary_crossentropy')
        self.DBN.fit(X, Y, batch_size = batch_size, epochs = int(0.25*epochs), verbose = 0)
        
        return RBM_weights
        
    def predict(self,X):
        return self.DBN.predict(X)
    






class db_xie(model_template):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        
        self.Output_A_train = self.Output_A_train.astype('float32')
        
        

    def train_method(self, l2_regulization = 0.01):
        # Multiple timesteps have to be flattened
        X_help = self.Input_path_train.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps]) * np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, (self.timesteps - len(X_help[i,j])):] =  X_help[i,j]
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        self.mean = np.nanmean(X, axis = 0, keepdims = True)
        X = X - self.mean
        X[np.isnan(X)] = 0
        
        # get parameters of random_forest_model
        self.xmin = np.min(X, 0, keepdims = True)
        self.xmax = np.max(X, 0, keepdims = True) 
        X = (X - self.xmin) / (1e-5 + self.xmax - self.xmin)
        
        Y = np.zeros((len(self.Output_A_train), 2))
        Y[:,0] = self.Output_A_train
        Y[:,1] = 1-self.Output_A_train
        
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        
        hidden_shape = []        
        
        # Initialize error terms for number layers
        E_num_layers_old = 10001
        E_num_layers = 10000 
        

        print('DBN')
        RBM_weights_old = []
        DBN_num_layers = None  
        
        while E_num_layers_old > E_num_layers:
            # Save old network and corresponding error
            E_num_layers_old = E_num_layers
            DBN_num_layers_old = DBN_num_layers
            
            # Add new layer to the DBN
            hidden_shape = hidden_shape + [1]
            
            print('DBN - {} hidden layers'.format(len(hidden_shape)))
            
            ## Create and train new network with optimal number of nodes in new layer
            # Initialize error terms for number nodes
            E_num_nodes_old = 10001
            E_num_nodes = 10000 
            
            # Initialize new network
            DBN_num_nodes = None
            RBM_weights_new = RBM_weights_old
            
            
            while E_num_nodes_old > E_num_nodes:                
                # Save old network and corresponding error
                E_num_nodes_old = E_num_nodes
                DBN_num_nodes_old = DBN_num_nodes
                RBM_weights = RBM_weights_new
                
                # create structure for new model by adding a node to last hidden layer
                hidden_shape[-1] = hidden_shape[-1] + 1
                structure = [input_dim] + hidden_shape + [output_dim]
                
                
                print('DBN - {} hidden layers - {} nodes in last layer'.format(len(hidden_shape),hidden_shape[-1]))
                # Create and train new network
                DBN_num_nodes = deep_belief_network(structure)
                RBM_weights_new = DBN_num_nodes.train(X, Y, RBM_weights_old, int(len(X)/10), 400, 500)  
                
                # Evaluate the error corresponding to the new network
                E_num_nodes = np.mean((Y - DBN_num_nodes.predict(X))**2)
                
                print('DBN - {} hidden layers - {} nodes in last layer - loss: {:0.4f}'.format(len(hidden_shape), hidden_shape[-1], E_num_nodes))
            # Use this new network
            DBN_num_layers = DBN_num_nodes_old
            RBM_weights_old = RBM_weights 
            hidden_shape[-1] = hidden_shape[-1] - 1
            # Get the error corresponding to the new network
            E_num_layers = E_num_nodes_old
            print('DBN - {} hidden layers -                       - loss: {:0.4f}'.format(len(hidden_shape), E_num_layers))
            
        
        
        self.DBN = DBN_num_layers_old
        hidden_shape = hidden_shape[:-1]
        
        self.structure = [input_dim] + hidden_shape + [output_dim]
    
        print('Final DBN structure: {} - loss: {:0.4f}'.format(self.structure,E_num_layers_old))
        
        DBN_weights = self.DBN.DBN.get_weights()
        
        self.weights_saved = [self.mean, self.xmin, self.xmax, self.structure, DBN_weights]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.xmin, self.xmax, self.structure, DBN_weights] = self.weights_saved
        
        self.DBN = deep_belief_network(self.structure) 
        
        Y = np.zeros((len(self.Output_A_train), 2))
        
        self.DBN.train(self.mean, Y[[0]], [], int(len(self.mean)/10), 0, 0)
        
        self.DBN.DBN.set_weights(DBN_weights)
        
        # set parameters
        
    def predict_method(self):
        # Multiple timesteps have to be flattened
        X_help = self.Input_path_test.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps])*np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, max(0,(self.timesteps - len(X_help[i,j]))):] =  X_help[i,j][max(0,(len(X_help[i,j]) - self.timesteps)):]
        X = X.reshape(X.shape[0], -1)
        X = X - self.mean
        # set missing values to zero
        X[np.isnan(X)] = 0
        X = (X - self.xmin) / (1e-5 + self.xmax - self.xmin)
        # make prediction
        Y_out = self.DBN.predict(X)
        
        Prob = Y_out[:,0]
        
        return [Prob]

    def check_input_names_method(self, names, train = True):
        if all(names == self.input_names_train):
            return True
        else:
            return False
     
    
    def get_output_type_class():
        # Logit model only produces binary outputs
        return 'binary'
    
    def get_name(self):
        return 'deep_belief'
        

        
        
        
        
        
    
        
        
        