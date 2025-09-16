import requests
import time
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

#-----------------
# list of cities
#----------------
cities =["Atlanta, GA", "Marietta, GA", "Kennesaw, GA","Duluth, GA", "Decatur, GA", "Alpharetta, GA", "Suwanee, GA", "Smyrna, GA", "Augusta, GA", "Gainesville, GA", "Dalton, GA", "Brookhaven, GA", "Roswell, GA"]

#-----------------
# Check if data already exists
#-----------------
if os.path.exists("commute_time_BF.csv") and os.path.exists("distance_BF.csv"):
    df_commute = pd.read_csv("commute_time_BF.csv", index_col=0)
    df_distance = pd.read_csv("distance_BF.csv", index_col=0)
else:
    #----------------
    # Creating empty dictionaries to build two matrices
    #----------------
    commute_time = {}
    distance = {}
    for i in range(len(cities)):
        origin = cities[i]
        commute_time[origin]={}
        distance[origin]= {}
        for j in range(len(cities)):
            destination = cities[j]
            if origin == destination:
                continue
            url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key=AIzaSyBopeGxSm80ys0jkzxUReY5l6h1SPSX1OE" 
            response = requests.get(url)
            time.sleep(1)
            data = response.json()
            element = data['rows'][0]['elements'][0]
            time_seconds = element['duration']['value']
            distance_meters = element['distance']['value']
            commute_time[origin][destination] = time_seconds
            distance[origin][destination]= distance_meters

    #-----------------
    # converting dictionaries into dataframes
    #----------------- 
    df_commute = pd.DataFrame(commute_time)
    df_distance = pd.DataFrame(distance)

    #------------------
    # Saving the data to csv file
    #------------------
    df_commute.to_csv("commute_time_BF.csv")
    df_distance.to_csv("distance_BF.csv")

#-------------------
# Normalizing time and distance data with equal priority
#-------------------
min_time = df_commute.min().min()
max_time = df_commute.max().max()

min_dist = df_distance.min().min()
max_dist = df_distance.max().max()

normalized_time = (df_commute - min_time) / (max_time - min_time)
normalized_dist = (df_distance - min_dist) / (max_dist - min_dist)

normalized_data = normalized_time + normalized_dist

#--------------------
# User input 
#--------------------
while True:
    destination_city = input("Enter the city of your workplace or school: ").strip()
    if destination_city not in cities:
        print("Enter a valid city of Georgia")
        continue
    
    all_valid = True
    source_cities = input("Enter the list of cities of your preference separated by semicolon: ").strip()
    source_cities_list = [city.strip() for city in source_cities.split(';')]
    for city in source_cities_list:
        if city not in cities:
            print(f"{city} is not a valid city of Georgia")
            all_valid = False

    if not all_valid:
        print("Please enter valid city names of Georgia")
        continue
    break

#-----------------------
# Building the adjacency list graph
#-----------------------
graph = {}
for origin in normalized_data.columns:
    graph[origin] = []
    for destination in normalized_data.index:
        if origin != destination:
            weight = normalized_data[origin][destination]
            graph[origin].append((destination, weight))

#--------------------
# Brute Force Shortest Path function 
#--------------------
def Brute_Force_shortest_path(graph, start, target):
    shortest_path = []
    shortest_distance = float('inf')

    def dfs(current_city, current_path, current_distance):
        nonlocal shortest_distance, shortest_path  
        if current_city == target:
            if current_distance < shortest_distance:
                shortest_distance = current_distance
                shortest_path = current_path
        else:
            for neighbor, weight in graph[current_city]:
                if neighbor not in current_path:
                    new_distance = current_distance + weight
                    dfs(neighbor, current_path + [neighbor], new_distance)

    dfs(start, [start], 0)
    return shortest_path, shortest_distance

#--------------------
# Collect all shortest paths from each source to destination using brute force
#--------------------
all_paths = []
target = destination_city

for start in source_cities_list:
    path, dist = Brute_Force_shortest_path(graph, start, target)
    if path:
        print(f"\nShortest path from {start} to {target}:")
        print(" → ".join(path))
        print(f"Distance: {dist:.4f}")
        all_paths.append(path)
    else:
        print(f"No path found from {start} to {target}")

#-------------
# Visualization
#-------------
# Build NetworkX graph
G = nx.DiGraph()
for origin, neighbors in graph.items():
    for dest, weight in neighbors:
        G.add_edge(origin, dest, weight=weight)

# Layout
pos = nx.spring_layout(G, seed=42)

#figure size
plt.figure(figsize=(32, 20)) 

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True, width=1)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

# Highlight all shortest paths 
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']  
seen_edges = {}  

for i, path in enumerate(all_paths):
    path_edges = list(zip(path[:-1], path[1:]))
    color = colors[i % len(colors)]
    
    for j, (u, v) in enumerate(path_edges):
        if (u, v) not in seen_edges:
            seen_edges[(u, v)] = 0
        rad = 0.2 * seen_edges[(u, v)]  
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            width=2.5,
            connectionstyle=f'arc3,rad={rad}',
            arrows=True
        )
        seen_edges[(u, v)] += 1

plt.title(f"Shortest paths to {target}")
plt.axis('off')
plt.savefig("shortest_paths_BF.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# Final Recommendation Section
# -----------------------------

# Load normalized cost of living data
file_path = '20 Cities.xlsx'
cost_data = {}
for city in cities:
    try:
        df_city_cost = pd.read_excel(file_path, sheet_name=city)
        total_cost = df_city_cost['Monthly Estimate (Family of 4)'].sum()
        cost_data[city] = total_cost
    except Exception as e:
        print(f"Error reading cost data for {city}: {e}")
        cost_data[city] = None

df_cost = pd.DataFrame.from_dict(cost_data, orient='index', columns=['TotalCost'])
min_cost = df_cost['TotalCost'].min()
max_cost = df_cost['TotalCost'].max()
df_cost['NormalizedCost'] = (df_cost['TotalCost'] - min_cost) / (max_cost - min_cost)

# Combine normalized commute cost (from brute force) and normalized cost of living
print("\n====================")
print("Final Recommendation")
print("====================")

best_city = None
best_total_cost = float('inf')

for start in source_cities_list:
    if start not in df_cost.index:
        continue

    path, commute_cost = Brute_Force_shortest_path(graph, start, destination_city)

    if commute_cost == float('inf') or commute_cost is None:
        continue

    living_cost = df_cost.loc[start, 'NormalizedCost']
    total_cost = 0.7*commute_cost + 0.3*living_cost

    print(f"{start}: Commute = {commute_cost:.4f}, Living = {living_cost:.4f}, Total = {total_cost:.4f}")

    if total_cost < best_total_cost:
        best_total_cost = total_cost
        best_city = start

if best_city:
    print(f"\n Best city to live in is: **{best_city}**")
    print(f"Optimal balance of commute and cost of living: {best_total_cost:.4f}")
else:
    print(" Could not determine the best city due to missing data.")
