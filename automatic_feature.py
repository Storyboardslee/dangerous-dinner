""" Code: To get a latent representation of a network graph
	Representation Learning for sc-RNA seq networks using Sparse Autoencoders
	Course: Biological Networks 	
	Author: Samyadeep Basu 															"""


#Libraries
import numpy as np
import pandas as pd 
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import roc_auc_score


#Function to read the network graph
def create_graph():
	#Read data
	df = pd.read_csv('data.txt',sep='\t', header=None)
	
	#Targets
	targets = df[2].values.tolist()

	#Regulators
	regulators = df[3].values.tolist()

	#Form a adjacency list - Regulators -> Target 
	adj_list = list(zip(regulators,targets))

	return adj_list

#Function to build classifier to give a score to the removed edges
def predict_edges(node, positive, negative, test, features, indexed_nodes):
	positive_indexes = [np.where(indexed_nodes==node)[0][0] for node in positive]

	negative_indexes = [np.where(indexed_nodes==node)[0][0] for node in negative]

	X_positive = features[positive_indexes][:]
	X_negative = features[negative_indexes][:]

	#Training Data
	X = np.append(X_positive, X_negative, axis=0)

	#Labels
	y_pos  = np.ones(len(X_positive))
	y_neg = np.ones(len(X_negative))


	return

#Function to build a local learning model
def build_model(adj_list, test_list, features, indexed_nodes):
	#Find Edges to inspect / build a model in the system
	regulators = np.unique(adj_list[:,0])

	#For each node which acts as a regulator build the model
	for node in regulators:
		#Find the nodes which is regulated by the current node
		positive_targets = [edge[1] for edge in adj_list if edge[0]==node]

		#Negative examples 
		negative_targets = []

		for reg in regulators:
			if reg not in positive_targets:
				negative_targets.append(reg)

		#Build classifier 
		predict_edges(node, positive_targets, negative_targets, test_list, features, indexed_nodes)

		break

	return

#Function to create a sorted indexed list 
def create_sorted_index(adjacency_list):
	sorted_list = []
	for node in adjacency_list:
		sorted_list.append(node[0])
		sorted_list.append(node[1])


	sorted_list = sorted(np.unique(np.array(sorted_list)))

	return sorted_list

#Function to create the model 
def create_model(adjacency_list, features):
	#Randomly Shuffle the list
	index = list(range(0,len(adjacency_list)))

	#Shuffle the index
	np.random.shuffle(index)

	#Adjacency List
	adjacency_list = np.array(adjacency_list)[index]

	#Prune Graph
	prune_amount = int(len(adjacency_list)/5)
	
	#Pruned Adjacency List
	pruned_list = adjacency_list[prune_amount:]

	#Testing List
	test_list = adjacency_list[:prune_amount]

	#Create a sorted list / Lookup table
	indexed_nodes = create_sorted_index(adjacency_list)

	#Build the model
	build_model(pruned_list, test_list, features, indexed_nodes)

	return

#Function to create feature vectors
def create_feature_vectors(adj_list):
	#Create networkX graph 
	G = nx.Graph(adj_list)

	#Extract adjacency matrix
	adj_matrix = nx.adjacency_matrix(G)

	#Dense matrix
	adj = adj_matrix.todense()

	return adj, G

############################################## Sparse AutoEncoder for representation learning for networks ################################

#Function to initialise weights
def initial_weights(shape):
	#Weight - Connections for the given dimensions - Sampled from a normal distribution
	weights = tf.random_normal(shape)

	#Wrap in tensorflow variable
	return tf.Variable(weights)

#Function to produce an encoding of the representation
def encode_graph(data, weights, biases):
	#Hidden Layer
	hidden = tf.matmul(data, weights) + biases

	#Activation function
	a_hidden = tf.nn.sigmoid(hidden)

	return a_hidden

#Function to produce a decoding of the graph from the learned representation
def decode_graph(hidden_data, weights, biases):
	#Output layer
	output = tf.matmul(hidden_data, weights) + biases

	#Activate
	a_output = tf.nn.sigmoid(output)

	return a_output

#Definition for KL Divergence 
def KL_divergence(rho, rho_hat):
	#KL Divergence functional value - for sparsity
	return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)


#Function to learn a good representation of the graph using sparse autoencoders
def learn_graph(matrix):
	#Input Nodes
	n_input = len(matrix)

	#Hidden Nodes
	n_hidden = 10

	#Parameters 
	alpha = 0.001  #Regularization parameters for weights
	beta = 3  #Regularization hyperparameter
	rho = 0.01 #Sparsity condition

	#Number of iterations
	n_iter = 4000

	#Create placeholder for data
	X = tf.placeholder("float", shape = [None, len(matrix)])

	#Weights for the initial -> hidden layer
	w1 = initial_weights((n_input, n_hidden))

	#Biases for the first hidden layer
	b1 = initial_weights((1,n_hidden))

	#Weights for the hidden layer -> final layer
	w2 = initial_weights((n_hidden, n_input))

	#Biases for the second hidden layer
	b2 = initial_weights((1,n_input))

	#Representation of the graph
	hidden = encode_graph(X, w1, b1)

	#Enforcement of sparsity
	rho_hat = tf.reduce_mean(hidden, axis=0)

	#Output Layer of the graph
	output = decode_graph(hidden, w2, b2)

	#Difference b/w actual data and learned data
	difference = output - X

	#KL Divergence term for sparsity
	kl_term = KL_divergence(rho,rho_hat)

	#Define the cost function - Mean square + regularization of weights + addition of sparsity
	cost = 0.5 * tf.reduce_mean(tf.reduce_sum(difference**2, axis=1)) + 0.5*alpha*(tf.reduce_mean(tf.nn.l2_loss(w1)) + tf.reduce_mean(tf.nn.l2_loss(w2))) + beta*(tf.reduce_sum(kl_term)) 

	#Define the loss function
	#train_step=tf.contrib.opt.ScipyOptimizerInterface(loss_, var_list=var_list, method='L-BFGS-B',   options={'maxiter': n_iter})

	#Training using Adam Optimizer
	training = tf.train.AdamOptimizer(0.001).minimize(cost)

	#Initialisation for Tensorflow
	init = tf.global_variables_initializer()

	#Start the session
	with tf.Session() as sess:
		sess.run(init)

		#Number of rounds
		n_rounds = n_iter

		#Batch size
		batch_size = 30

		for i in range(n_rounds):
			#Generate random batch indexes
			batch_indexes = np.random.randint(len(matrix),size=batch_size)

			#Creation of mini batch
			input_data = matrix[batch_indexes][:]

			#Run the training scheme developed
			sess.run(training, feed_dict={X:input_data})

			#Get the loss value
			print(sess.run(cost, feed_dict={X:input_data}))



		#Get the weights for the hidden layer
		w_hidden = sess.run(w1)
		b_hidden = sess.run(b1)


		#Learned embedded graph representation
		embedding = np.matmul(matrix,w_hidden) + b_hidden
		

	return embedding

###########################################################################################################################################

#Function to compute edge features
def compute_edge_features(positive_edges, negative_edges, graph_embedding):
	positive_features = np.array([np.multiply(graph_embedding[edge[0]],graph_embedding[edge[1]])[0] for edge in positive_edges])

	positive_edges = np.reshape(positive_features, (positive_features.shape[0],positive_features.shape[2]))

	negative_features = np.array([np.multiply(graph_embedding[edge[0]],graph_embedding[edge[1]])[0] for edge in negative_edges])

	negative_edges = np.reshape(negative_features, (negative_features.shape[0],negative_features.shape[2]))

	y_positive = np.ones(len(positive_edges))
	y_negative = np.zeros(len(negative_edges))


	X = np.append(positive_edges,negative_edges,axis=0)
	y = np.append(y_positive, y_negative, axis=0)

	return X,y


#Function to perform the classification
def classify_edges(X,y):
	#Shuffle the points 
	indexes = np.arange(len(X))

	#Shuffle the indices
	np.random.shuffle(indexes)

	X = X[indexes]
	y = y[indexes]

	#Split for cross validation
	kf = KFold(n_splits=5)

	scores = 0

	#Learning and evaluation for the model
	for train_index, test_index in kf.split(X):
		#Training and Testing
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		#Initialise SVM
		clf = svm.SVR(gamma='scale', C=0.00001)

		#Fit model
		clf.fit(X_train, y_train)

		#Predictions from the learning model
		predictions = clf.predict(X_test)

		#ROC Curve
		scores += roc_auc_score(y_test, predictions)
	

	final_score = float(scores)/5

	return final_score

#Learning Model using Network Embeddings
def learning_model(graph_embedding, adjacency_list):
	#Remove self links
	adj = [edge for edge in adjacency_list if edge[0]!=edge[1]]

	#Create Indexes 
	index = np.array(create_sorted_index(adj))

	#Update the positive edges
	positive_edges = [(np.where(index == edge[0])[0][0],np.where(index==edge[1])[0][0]) for edge in adj]

	complete_edges = []

	#Construction of Negative edges
	for i in range(0,len(index)):
		for j in range(0,len(index)):
			if i!=j:
				complete_edges.append((i,j))

	#Negative Edges
	negative_edges = [edge for edge in complete_edges if edge not in positive_edges]

	#Training Data
	X, y = compute_edge_features(positive_edges, negative_edges, graph_embedding)

	classify_edges(X, y)

	return

#Function to visualise the embedding
def visualise_embedding(graph_embedding):
	#X-coordinate
	X = list(graph_embedding[:,0])

	#Y-coordinate
	y = list(graph_embedding[:,1])

	plt.scatter(X,y)
	plt.show()
	return

#Function - Main
def main():
	#Form the adjacency list
	adjacency_list = create_graph()

	#Create feature vectors
	adj, G = create_feature_vectors(adjacency_list)

	#Creation of the graph embedding
	graph_embedding = learn_graph(adj)

	#Visualisation for the embedding
	#visualise_embedding(graph_embedding)

	#Link Prediction Scheme using network embeddings
	final_auc_score = learning_model(graph_embedding, adjacency_list)

	print(final_auc_score)


	return


main()


