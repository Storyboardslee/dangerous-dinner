import os
import sys
import argparse
import networkx as nx
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',type = str,required = True)
parser.add_argument('-o','--output',type = str,required = True)
args = parser.parse_args(sys.argv[1:])

G = nx.DiGraph()
df = pd.read_csv(args.input,sep = '\t',header =None)
df = df[[0,1]]
edge_list = []
for index, row  in df.iterrows():
    edge_list.append((row[0],row[1]))
G.add_edges_from(edge_list)
A = nx.adjacency_matrix(G)
AM = nx.to_numpy_matrix(G)
np.savetxt(args.output, AM, delimiter=',', fmt='%d')  