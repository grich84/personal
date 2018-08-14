#Author Garrett Richardson
#For this project we used multiple algorithms against each other to determine the largest suspects on multiple flights
#We ran heuristic, brute force, and approximation algorithms. The best one we found was to create a threshold that will 
#eliminate all individuals who should not be considered a suspect and then move on from that smaller data set creating a 
#smaller time complexity


import math
import numpy
import csv
import networkx as nx
import matplotlib.pyplot as plt
import random
import time

# Load the csv file into a large 2D array
def loadArray(filepath):
    list = []
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            list2 = numpy.array(row)
            list.append(list2.astype(float))
    return (list)

# This function is going to be used to reduce the number of suspects that are
# going to be looked at

def getSuspectList(flightList):
    count = [0 for i in range(0,300)] # Storing the total number of time each person flies.
    for flight in flightList:
        for person in flight: # Increase the flight count for the person
            count[int(person) - 1] = count[int(person) - 1] + 1
   
    return (count) # Have continued to use this function so we know how many flights each suspect
                   # has taken.

# Finds the highest weight edge for a given node.
def getMaxWeight(G, node):
    maxWeight = 0
    for i in len(G.edges(node)):
        currentWeight = G[node][G.edges(node)[i][1]]['weight']
        if currentWeight > maxWeight:
            maxWeight = currentWeight
    return maxWeight

# Print the nodes which are connected, with the percentage of how many flights
# they shared, vs. their total flights.
def printEdgesWithWeights(G, totalFlights):
    for u,v in [(u, v) for (u, v) in G.edges()]:
        u_percent = round((G[u][v]['weight'] / totalFlights[int(u - 1)]) * 100)
        v_percent = round((G[u][v]['weight'] / totalFlights[int(v - 1)]) * 100)
        print(u_percent,"% of", u,"'s ->", v_percent,"% of", v,"'s with a weight of:",G[u][v]['weight'])

# A function to delete irrelevant nodes from a tree based on the given
# threshold, and percentage of total flights
def delIrrelevant(G,threshold,totalFlights):
    nodes_to_remove = []
    for node in G.nodes():
        safe = False
        edges_to_remove = []
        for edge in [(v) for (u, v) in G.edges(node)]:
            #percent = round((G[node][edge]['weight']/totalFlights[int(node - 1)]) * 100)
            # If the weight of the edge is less than the threshold, then we
            # don't care for it.
            if G[node][edge]['weight'] > threshold: # IF WE WANT TO IMPLEMENT THE PERCENTAGE GUY, ITS HERE
                safe = True
            else:
                edges_to_remove.append([node, edge])
        G.remove_edges_from(edges_to_remove)
        if safe == False:
            nodes_to_remove.append(node)# add current node to list that will be deleted
    G.remove_nodes_from(nodes_to_remove)
    return G

# Do not look at every flight to construct the initial graph.
def notAllFlights(flightList,G,threshold,ratio,totalFlights):
    for flightnum in range(0,round(len(flightList) * ratio)): # Traverse through all of the flights in the flightlist
        flight = flightList[flightnum]
        for i in range(0,len(flight)): # i the index, flight[i] is ID (node) of suspect
            for j in range(i + 1,len(flight)): # j is the index, flight[j] is the ID (node) of supect
                if G.has_edge(flight[i], flight[j]):
                    # we added this one before, just increase the weight by
                    # one
                    G[flight[i]][flight[j]]['weight'] += 1
                else:
                    # new edge.  add with weight=1
                    G.add_edge(flight[i], flight[j], weight=1)

    G = delIrrelevant(G,threshold,totalFlights)

# Do not look at all people in every flight to construct the graph.
def notAllPeople(flightList,G,threshold,ratio,totalFlights):
    for flight in flightList: # Traverse through all of the flights in the flightlist
        where_to_go = int(len(flight) * ratio)
        random.shuffle(flight) # Shuffle the order of the flight so that we're not always looking at the
                               # lowest numbers
        for i in range(0,where_to_go): # i the index, flight[i] is ID (node) of suspect
            for j in range(i + 1,where_to_go): # j is the index, flight[j] is the ID (node) of supect
                if G.has_edge(flight[i], flight[j]):
                    # we added this one before, just increase the weight by one
                    G[flight[i]][flight[j]]['weight'] += 1
                else:
                    # new edge.  add with weight=1
                    G.add_edge(flight[i], flight[j], weight=1)

    G = delIrrelevant(G,threshold,totalFlights)

# Do not look at all flights, and do not look at all people.
def notAllPeopleFlights(flightList,G,threshold,ratiof,ratiop,totalFlights):
    for flightnum in range(0,int(len(flightList) * ratiof)): # Traverse through all of the flights in the flightlist
        flight = flightList[flightnum]
        where_to_go = int(len(flight) * ratiop)
        random.shuffle(flight)
        for i in range(0,where_to_go): # i the index, flight[i] is ID (node) of suspect
            for j in range(i + 1,where_to_go): # j is the index, flight[j] is the ID (node) of supect
                if G.has_edge(flight[i], flight[j]):
                    # we added this one before, just increase the weight by one
                    G[flight[i]][flight[j]]['weight'] += 1
                else:
                    # new edge.  add with weight=1
                    G.add_edge(flight[i], flight[j], weight=1)

    G = delIrrelevant(G,threshold,totalFlights)

# This function stores into the graph G the entire suspect list with all edges
# and vertices, then deletes the connections not above the threshold.
def bruteForce(flightList,G, threshold,totalFlights):
    for flight in flightList: # Traverse through all of the flights in the flightlist
        for i in range(0,len(flight)): # i the index, flight[i] is ID (node) of suspect
            for j in range(i + 1,len(flight)): # j is the index, flight[j] is the ID (node) of supect
                if G.has_edge(flight[i], flight[j]):
                    # we added this one before, just increase the weight by one
                    G[flight[i]][flight[j]]['weight'] += 1
                else:
                    # new edge.  add with weight=1
                    G.add_edge(flight[i], flight[j], weight=1)

    G = delIrrelevant(G,threshold,totalFlights)

# A brute force algorithm to collect all information about one node.
def byFlightNode(flightList, G, node, threshold,totalFlights):
    for flight in flightList: # Traverse through all of the flights in the flightlist
        if node in flight:
            for i in range(0,len(flight)): # i the index, flight[i] is ID (node) of suspect
                if G.has_edge(flight[i], node):
                    # we added this one before, just increase the weight by one
                    G[flight[i]][node]['weight'] += 1
                else:
                    # new edge.  add with weight=1
                    G.add_edge(flight[i], node, weight=1)

    # Delete connections which are not above the threshold
    for v in [(v) for (u,v) in G.edges(node)]:
        if G[node][v]['weight'] < threshold:
            G.remove_node(v)

# Main
# ========================================================================================================================================
# Create an undirected graph, and add all 300 nodes.
G = nx.Graph()
G.add_nodes_from(range(1,301))

# Ask which data set they want to use
original = 'lists.csv'
new = 'newlists.csv'
flightList = loadArray(new)
totalFlights = getSuspectList(flightList)

# Ask threshold
# NOTE: Useful results are found when using a threshold of at least 30 for the
# original data set, and 53 when using the new data set
threshold = 34

# Timing begin
begin = time.time()

# Which algorithm should you use?
# Want the correct answer?
#bruteForce(flightList,G,threshold,totalFlights)

# Want to analyze a single node?
#node = 88 # The node to be analyzed
#byFlightNode(flightList,G,node,threshold,totalFlights)

# Not all flights, not all people, not all of both
# NOTE: As the ratio is reduced, the threshold will also have to be reduced to
# account for the fact that not all data points will be considered
# NOTE: To use euler's it is math.e
# RATIOS: 2/e, 0.75, 0.6
ratiof = 0.6 # Use if not analyzing all flights
#ratiop = 0.75 # Use if not alayzing all people
notAllFlights(flightList,G,threshold,ratiof,totalFlights)
#notAllPeople(flightList,G,threshold,ratiop,totalFlights)
#notAllPeopleFlights(flightList,G,threshold,ratiof,ratiop,totalFlights)

printEdgesWithWeights(G,totalFlights)

# Timing end, round to 2 decimal places.
print("Total time taken: ",round((time.time() - begin),2)," seconds.")

pos = nx.circular_layout(G)  # Layout the graph is a circular orientation

# Draw nodes.
nx.draw_networkx_nodes(G, pos, node_size=800)

# Draw edges.
nx.draw_networkx_edges(G, pos, width=2)

# Draw labels for edges and nodes.
nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif', font_color='black')
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

# Show the plot.
plt.axis('off')
plt.show()
