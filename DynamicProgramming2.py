import requests
import time
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

#-----------------
# Load list of cities
#----------------
df = pd.read_excel("cities.xlsx")  
cities = df.iloc[:, 0].dropna().tolist() 

#-----------------
# Load or create distance and commute data
#-----------------
if os.path.exists("commute_time.csv") and os.path.exists("distance.csv"):
    df_commute = pd.read_csv("commute_time.csv", index_col=0)
    df_distance = pd.read_csv("distance.csv", index_col=0)
else:
    commute_time = {}
    distance = {}
    for i in range(len(cities)):
        origin = cities[i]
        commute_time[origin] = {}
        distance[origin] = {}
        for j in range(len(cities)):
            destination = cities[j]
            if origin == destination:
                continue
            url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key=YOUR_API_KEY"
            response = requests.get(url)
            time.sleep(1)
            data = response.json()
            element = data['rows'][0]['elements'][0]
            time_seconds = element['duration']['value']
            distance_meters = element['distance']['value']
            commute_time[origin][destination] = time_seconds
            distance[origin][destination] = distance_meters

    df_commute = pd.DataFrame(commute_time)
    df_distance = pd.DataFrame(distance)
    df_commute.to_csv("commute_time.csv")
    df_distance.to_csv("distance.csv")

#-------------------
# Normalize time and distance
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
    
    source_cities = input("Enter your preferred cities separated by semicolon: ").strip()
    source_cities_list = [city.strip() for city in source_cities.split(';')]
    
    if all(city in cities for city in source_cities_list):
        break
    print("Some cities are invalid. Please re-enter.")

#-----------------------
# Build graph from normalized data
#-----------------------
graph = {}
for origin in normalized_data.columns:
   graph[origin] = []
   for destination in normalized_data.index:
       if origin != destination:
            weight = normalized_data[origin][destination]
            graph[origin].append((destination, weight))

#-----------------------------
#Dynamic programming
#-----------------------------
distance_between = {}
next_city = {}

# Initialize distances
for i in cities:
    for j in cities:
        if i == j:
            distance_between[(i, j)] = 0
            next_city[(i, j)] = None
        else: 
            if i in normalized_data.columns and j in normalized_data.index:
                weight = normalized_data.at[j, i]
            else:
                weight = float('inf')

            distance_between[(i, j)] = weight
            if weight != float('inf'):
                next_city[(i, j)] = j
            else : 
                None

for k in cities:
    for i in cities:
        for j in cities:
            if distance_between[(i, k)] + distance_between[(k, j)] < distance_between[(i, j)]:
                distance_between[(i, j)] = distance_between[(i, k)] + distance_between[(k, j)]
                next_city[(i, j)] = next_city[(i, k)]

def reconstruct_path(source, target):
    if next_city[(source, target)] is None:
        return []
    path = [source]
    while source != target:
        source = next_city.get((source, target))
        if source is None:
            return []
        path.append(source)
    return path

#-----------------------------
# Compute paths from source cities
#-----------------------------
all_paths = []
for source in source_cities_list:
    cost = distance_between.get((source, destination_city), float('inf'))
    path = reconstruct_path(source, destination_city)
    if not path:
        print(f"No path from {source} to {destination_city}")
        continue
    print(f"\nShortest path from {source} to {destination_city}:")
    print(" → ".join(path))
    print(f"Distance: {cost:.4f}")
    all_paths.append(path)

# ---------------------
# Build NetworkX graph
# ---------------------
G = nx.DiGraph()
for origin, neighbors in graph.items():
    for dest, weight in neighbors:
        G.add_edge(origin, dest, weight=weight)

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(32, 20))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1600)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=2)
nx.draw_networkx_labels(G, pos, font_size=14)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

# Highlight all shortest paths 
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'darkgreen', 'magenta', 'black']
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
            width=5.5,
            connectionstyle=f'arc3,rad={rad}',
            arrows=True,
            arrowstyle='-|>',
            min_target_margin=17 
        )
        seen_edges[(u, v)] += 1

plt.title(f"Shortest paths to {destination_city}", fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("shortest_paths_DP.png", dpi=600, bbox_inches='tight')
plt.show()

# -----------------------------
# Final Recommendation
# -----------------------------
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

# Recommendation based on commute + living cost
print("\n====================")
print("Final Recommendation")
print("====================")

best_city = None
best_total_cost = float('inf')

for start in source_cities_list:
    if start not in df_cost.index:
        continue
    commute_cost = distance_between.get((start, destination_city), float('inf'))
    if pd.isna(commute_cost) or commute_cost == float('inf'):
        continue
    living_cost = df_cost.loc[start, 'NormalizedCost']
    total_cost = 0.7 * commute_cost + 0.3 * living_cost
    print(f"{start}: Commute = {commute_cost:.4f}, Living = {living_cost:.4f}, Total = {total_cost:.4f}")

    if total_cost < best_total_cost:
        best_total_cost = total_cost
        best_city = start

if best_city:
    print(f"\n Best city to live in is: **{best_city}**")
    print(f"Optimal balance of commute and cost of living: {best_total_cost:.4f}")
else:
    print(" Could not determine the best city due to missing data.")
