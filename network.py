""" Code: To get a latent representation of a network graph
	Course: Biological Networks 												"""


#Libraries
import numpy as np
import pandas as pd 


#Function to read the network graph
def create_graph():
	#Read data
	df = pd.read_csv('data.txt',sep='\t', header=None)
	
	#Targets
	targets = df[2].values.tolist()

	#Regulators
	regulators = df[3].values.tolist()

	#Form a adjacency list - Regulators, Target 
	adj_list = list(zip(regulators,targets))

	return adj_list

#Function to build classifier
def build_classifier():
	return

#Function to build a local learning model
def build_model(adj_list, test_list):
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
		classifier()
		
		break

	return

#Function to create the model 
def create_model(adjacency_list):
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

	#Build the model
	build_model(pruned_list,test_list)

	return

#Function - Main
def main():
	#Form the adjacency list
	adjacency_list = create_graph()

	create_model(adjacency_list)
	return


main()


