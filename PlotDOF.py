import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.spatial import KDTree
from collections import Counter
from shapely.geometry import Point, Polygon
import random


def dms_to_dd(d, m, s):
    """ Convert degree, minute, second to decimal degree. """
    return d + (m / 60.0) + (s / 3600.0)

def polar_to_cartesian(polar_array, error_percent, error_percent_ang, error_stdev, error_stdev_angle, Error):
    polar_array = np.transpose(polar_array)
    indexes = polar_array[0]
    distances = polar_array[1]
    angles = polar_array[2]
    if Error == True:
        rand_dist = np.random.normal(0, error_stdev, np.size(distances))
        rand_ang = np.random.normal(0, error_stdev_angle, np.size(angles))
        x = (distances + rand_dist) * np.cos(angles + rand_ang)
        y = (distances + rand_dist) * np.sin(angles + rand_ang)
    
    else:
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
    
    cartesian_array = np.vstack((indexes, y, x))
    
    return cartesian_array

def calculate_distances_angles(cartesian_array):

    print(cartesian_array.shape)
    # print(cartesian_array.shape[1])

    num_points = cartesian_array.shape[1]
    
    # Initialize distances and angles arrays
    distances = np.zeros((num_points, num_points))
    angles = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                # Calculate differences in coordinates
                dx = cartesian_array[0, i] - cartesian_array[0, j]
                dy = cartesian_array[1, i] - cartesian_array[1, j]
                
                # Calculate distances
                distances[i, j] = np.sqrt(dx**2 + dy**2)
                
                # Calculate angles
                angles[i, j] = np.arctan2(dy, dx)
    print(f'dists:', distances)
    print(f'angles:', angles)            
    return distances, angles

def parse_dms(dms):
    parts = dms
    degrees = float(parts[0])
    minutes = float(parts[1].split("'")[0])
    seconds = float(parts[1].split("'")[1].replace('"', ''))
    return dms_to_dd(degrees, minutes, seconds)

def read_lat_lon(filename, lat, lon, agl, msl, MIN_AGL):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[4:]:
            data = line.split()
            if len(data) >= 8:  
                # print(line)
                x = 5
                try:
                    int(data[x])
                except ValueError:
                    x=6
                    try:
                        int(data[x])
                    except ValueError:
                        x=7
                        try:
                            int(data[x])
                        except ValueError:
                            x=8
                lat_deg = np.float32(str(data[x]))
                lat_min = np.float32(str(data[x+1]))
                lat_sec = str(data[x+2])
                lat_NS = str(lat_sec[5:6:1])
                # print(lat_NS)
                lat_sec= np.float32(str(lat_sec[0:5:1]))
                # print(lat_sec)
                lon_deg = np.float32(str(data[x+3]))
                lon_min = np.float32(str(data[x+4]))
                lon_sec = str(data[x+5])
                lon_EW = str(lon_sec[5:6:1])
                # print(lon_EW)
                lon_sec= np.float32(str(lon_sec[0:5:1]))      
                # print(lon_sec)

                agl_str = np.float32(str(data[x+8]))
                msl_str = np.float32(str(data[x+9]))
                if agl_str >= MIN_AGL:
                    try:
                        if lat_NS == 'N':
                            lat.append(dms_to_dd(lat_deg, lat_min, lat_sec))
                        elif lat_NS == 'S':
                            lat.append(-1 * dms_to_dd(lat_deg, lat_min, lat_sec))
                        
                        if lon_EW == 'E':
                            lon.append(dms_to_dd(lon_deg, lon_min, lon_sec))
                        elif lon_EW == 'W':
                            lon.append(-1 * dms_to_dd(lon_deg, lon_min, lon_sec))
                        agl.append(agl_str)
                        msl.append(msl_str)
                    except IndexError:
                        continue  # Ignore lines that don't have enough data
    return lat, lon, agl, msl

def plot_coordinates(lat, lon, agl, SB1, SB2, x2, y2, x3, y3, x4, y4, colour):
    m = Basemap(projection='merc', llcrnrlat=min(lat)-1, urcrnrlat=max(lat)+1,
                llcrnrlon=min(lon)-1, urcrnrlon=max(lon)+1, resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='aqua')

    x, y = m(lon, lat)
    x1, y1 = m(SB2, SB1)
    x2, y2 = m(y2, x2)
    x3, y3 = m(y3, x3)
    x4, y4 = m(y4, x4)
    
    m.scatter(x, y, color=colour, marker='o', label='Points of Interest')
    m.scatter(x1, y1, color='red', marker='x', label='Drone')
    m.scatter(x2, y2, color='yellow', marker='x', label='Simulated points')
    
    m.scatter(x3, y3, color='green', marker='o', label='Matched points')
    m.scatter(x4, y4, color='green', marker='x', label='Nearest points')

def match_indexes_between_kdtrees(large_tree, small_trees, threshold):
    matched_indexes = []

    for small_tree in small_trees:
        distances, indices = large_tree.query(small_tree.data, k=1)
        matched_indices = np.where(distances <= threshold)[0]
        matched_indexes.append(indices[matched_indices])

    return matched_indexes

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def max_idx(index_list):
    index_counter = Counter(index_list)
    most_common = index_counter.most_common(1)
    print(f'mc{most_common}')
    if most_common != []:
        return most_common[0][0]
    else:
        return None

def generate_random_points_in_polygon(latitudes, longitudes, num_points):
    # Create a list of (latitude, longitude) tuples that define the polygon
    polygon_points = list(zip(longitudes, latitudes))
    
    # Create a polygon using Shapely
    polygon = Polygon(polygon_points)
    
    # Get the bounding box of the polygon
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    # print(polygon.boundary)
    # Generate random points within the polygon
    random_points = []
    while len(random_points) < num_points:
        # Generate a random point within the bounding box
        rand_lat = random.uniform(min_lat, max_lat)
        rand_lon = random.uniform(min_lon, max_lon)
        random_point = Point(rand_lon, rand_lat)
        
        # Check if the point is inside the polygon
        if polygon.contains(random_point):
            random_points.append((rand_lat, rand_lon))
    
    return random_points

def plot_coordinates_2(points):
    m = Basemap(projection='merc', llcrnrlat=min(lat)-1, urcrnrlat=max(lat)+1,
                llcrnrlon=min(lon)-1, urcrnrlon=max(lon)+1, resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='aqua')

    x, x2, x3, x4, x5 = [], [], [], [], []
    y, y2, y3, y4, y5 = [], [], [], [], []
    print(points)

    for i in points:
        if i[1] <= 1:
            x.append(i[0][1])
            y.append(i[0][0])
        elif 1 < i[1] <= 2:
            x2.append(i[0][1])
            y2.append(i[0][0])
        elif 2 < i[1] <= 3:
            x3.append(i[0][1])
            y3.append(i[0][0])
        elif 3 < i[1] <= 4:
            x4.append(i[0][1])
            y4.append(i[0][0])
        elif 4 < i[1]:
            x5.append(i[0][1])
            y5.append(i[0][0])

    x, y = m(x, y)
    x2, y2 = m(x2, y2)
    x3, y3 = m(x3, y3)
    x4, y4 = m(x4, y4)
    x5, y5 = m(x5, y5)

    m.scatter(x, y, color='red', marker='x', label='Drone')
    m.scatter(x2, y2, color='orange', marker='x', label='Drone')
    m.scatter(x3, y3, color='yellow', marker='x', label='Drone')
    m.scatter(x4, y4, color='green', marker='x', label='Drone')
    m.scatter(x5, y5, color='blue', marker='x', label='Drone')
    
    plt.show()
    
class Line_of_sight:
    def __init__(self, name):
        self.name = name
        self.Lookup = []
        self.index_a = []
        self.index_b = []
        self.distance = []
        self.angle = []
        self.above_gl = []
        self.above_msl = []

class Dict_maker:
    def __init__(self, name):
        self.name = name
        self.Lookup = []
        self.index_a = []
        self.index_b = []
        self.distance = []
        self.angle = []

class KD_calcs:
    def __init__(self, name):
        self.name = name
        self.Lookup = []
        self.index_a = []
        self.index_b = []
        self.distance = []
        self.angle = []

    def dist_calc(self, array1, array2):
        dist = math.sqrt(((array1[0] - array2[0]) ** 2) + ((array1[1] - array2[1]) ** 2))
        return dist
    
    def angle_calc(self, array1, array2):
        angle = math.atan2((array1[0] - array2[0]),(array1[1] - array2[1]))
        return angle

    def KD_Tree(self, nparray):
        Tree = KDTree(nparray)
        return Tree

    def create_KD_Tree(self, lat, lon, agl):
        lat = np.array(lat)
        lon = np.array(lon)
        agl = np.array(agl)
        lat_min = np.min(lat)
        lon_min = np.min(lon)
        lat_max = np.max(lat)
        lon_max = np.max(lon)
        agl_min = np.min(agl)
        agl_max = np.max(agl)
        KDarray = np.column_stack((lat, lon))
        Tree = KDTree(KDarray)
        x = 0
        for row in (KDarray):
            index = Tree.query(row, k=6)
            # print(f'index:', index)
            index = index[1][1:]
            # print((index))
            # print(f'row:',row)
            
            for i in index:
                # print(f'array', KDarray[i])
                dist = indexes.dist_calc(KDarray[i], row)
                angle = indexes.angle_calc(KDarray[i], row)
                # print(dist)
                
                self.index_a.append(x)
                self.index_b.append(i)
                self.distance.append(dist)
                self.angle.append(angle)
                # print(self.index_a)
                # print(self.index_b)
                # print(self.distance)
                # print(self.angle)
            x = x+1
        return self, self.index_a, self.index_b, self.distance, self.angle, lat_min, lat_max, lon_min, lon_max, agl_min, agl_max, Tree, KDarray

    def Lookup_Table(self, index_a, index_b, distance, angle):
        lookup_table = {}
        print(f'length of table:{len(index_a)}')
        for i in range(len(index_a)):
            indexa = index_a[i]
            indexb = index_b[i]
            distance_ab = distance[i]
            angle_ab = angle[i]

            if distance_ab != 0 and distance_ab or angle_ab not in lookup_table:
                lookup_table[distance_ab, angle_ab] = []
            # if angle_ab not in lookup_table:
                # lookup_table[angle_ab] = []
                # print(angle_ab)

            lookup_table[distance_ab, angle_ab].append((indexa, indexb))
            # lookup_table[angle_ab].append((indexa, indexb))

        return lookup_table

    def index_matching(self, lookup_table, target_dist, threshold):
        matching_indexes = []

        for distance_ab in lookup_table.keys():
            if (target_dist - threshold) <= distance_ab <= (target_dist + threshold):
                matching_indexes.extend(lookup_table[distance_ab])
        # print(f'matches', matching_indexes)
        return matching_indexes

def random_point_within_radius(random_points, kd_tree, radius):
    point_counts = []
    points_1 = []
    # Iterate over each random point and use the KDTree to find nearby points
    for point in random_points:
        lat, lon = point
        
        # Query the KDTree to find all points within the given radius of the random point
        indices = kd_tree.query_ball_point([lat, lon], r=radius)
        
        if len(indices) <= 5:
            if len(indices) >=4:
                nearest_idx = kd_tree.query([lat, lon], k=1)
                nearest = kd_tree.data[nearest_idx[1]]
                nearest_top3 = kd_tree.query(nearest, k=3)
            
        points_1.append([point, len(indices)])

        point_counts.append(len(indices))
    
    return point_counts, points_1

def plot_histogram(counts):
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    std_dev = np.std(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)

    # Print statistical metrics
    print(f"Mean: {mean_count}")
    print(f"Median: {median_count}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Min: {min_count}")
    print(f"Max: {max_count}")

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    
    # Add titles and labels
    plt.title('Histogram of Points within Radius')
    plt.xlabel('Number of Points within Radius')
    plt.ylabel('Frequency')

    # Display the histogram
    plt.show()

def DOF_heatmap(lat, lon):
    plt.figure(figsize=(10, 8))
    m = Basemap(projection='merc',
            llcrnrlat=min_lat_nd-0.25, urcrnrlat=max_lat_nd+0.25,
            llcrnrlon=min_lon_nd-0.5, urcrnrlon=max_lon_nd+0.5, 
            resolution='i')
    m = Basemap(projection='merc',
            llcrnrlat=min(lat), urcrnrlat=max(lat),
            llcrnrlon=min(lon), urcrnrlon=max(lon), 
            resolution='i')

    m.drawcoastlines()
    m.drawcountries(color='red')
    m.drawstates(color='red')
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='blue')


    # Convert to map projection coordinates
    x, y = m(lon, lat)

    # Create a heat map of point density
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=500)

    

    # Calculate the meshgrid for the heatmap
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Plot
    m.pcolormesh(xedges, yedges, heatmap.T, cmap='hot', shading='auto', vmin=0, vmax=10)
    
    
    # Add color bar
    plt.colorbar(label='Density', shrink=0.6)

    # Show the plot
    plt.title('Obstacle Density')
    plt.show()

def main(indexes, index_a, index_b, distance, angle, lat_min, lat_max, lon_min, lon_max, agl_min, agl_max, Tree, KDarray, lookup_table, LofS):

    Error = True
    error_percent = 0 # 0.002
    error_stdev = 0 #0.0005 ### Meters
    error_percent_angle = 0 #0.005
    error_stdev_angle = 0 #0.00025 ### Radians
    
    #Create KD Tree of Database
    # indexes = KD_calcs('distances')
    # indexes, index_a, index_b, distance, angle, lat_min, lat_max, lon_min, lon_max, agl_min, agl_max, Tree, KDarray = KD_calcs.create_KD_Tree(indexes, lat, lon, agl)

    # lookup_table = KD_calcs.Lookup_Table(indexes, index_a, index_b, distance, angle)

    # print((lookup_table))

    index_a = np.asarray(index_a, dtype=int)
    index_b = np.asarray(index_b, dtype=int)
    distance = np.array(distance)
    
    array = np.column_stack((index_a, index_b, distance))
    # print(array)

    SB_index = Tree.query_ball_point(SB_Point, r = LofS)
    # print(SB_index)
    SB_dists = []
    SB_angles = []
    SB_Match_points = []
    z = 0
    for i in SB_index:
        SB_dist = KD_calcs.dist_calc(indexes, array1=KDarray[i], array2=SB_Point)
        SB_angle = KD_calcs.angle_calc(indexes, array1=KDarray[i], array2=SB_Point)
        SB_Match_points.append(i)
        # print(SB_angle)
        z = z+1
        SB_dists.append(SB_dist)
        SB_angles.append(SB_angle)
        # SB_Match = KD_calcs.index_matching(indexes, lookup_table, SB_dist, (SB_dist/1000))
        # print(np.array(SB_Match).size)
        # Match_points.append(KDarray[SB_Match[0]])
    print(f'nearby points:', z)
    SB_Match_points = np.array(SB_Match_points)
    # print(SB_Match_points)
    SB_dists = np.array(SB_dists)
    SB_angles = np.array(SB_angles)

    SB_Match_Data = np.column_stack((SB_Match_points, SB_dists, SB_angles))
    # print(SB_Match_Data)

    SB_Match_Data = polar_to_cartesian(SB_Match_Data, error_percent, error_percent_angle, error_stdev, error_stdev_angle, Error)
    # print(f'cartesian:',SB_Match_Data[1])
    # SB_Match_Distances, SB_Match_Angles = calculate_distances_angles(SB_Match_Data)

    ### Old error addition, boxes of error around point, probably don't use
    # rand_lat = np.random.standard_normal(size=(SB_Match_Data[1].size))
    # print (rand_lat)
    # rand_lon = np.random.standard_normal(size=(SB_Match_Data[2].size))
    # print (rand_lon)
    # SB_Sim_Error_lat = rand_lat * (SB_Match_Data[1] * error_percent)
    # SB_Sim_Error_lon = rand_lon * (SB_Match_Data[2] * error_percent)
    

    Match_points_lat = SB_Match_Data[1] #+ SB_Sim_Error_lat
    Match_points_lon = SB_Match_Data[2] #+ SB_Sim_Error_lon
    
    SB_Sim_Points = np.column_stack((Match_points_lat, Match_points_lon))

    Sim_dexes = KD_calcs('SB_sim')
    Sim_Tree = KD_calcs.KD_Tree(Sim_dexes, SB_Sim_Points)
    Nearest_Points = Sim_Tree.query((0.0, 0.0), k=4) #r= half of search radius above. can be anything technically
    print(f'Nearest Points:{Nearest_Points}')

    Nearest_Points_WF = Tree.query(SB_Point, k=4)
    
    # Number of nearest neighbors to consider
    nearest_neighbors = 4
    points = []

    # Iterate through the indices of nearest points
    for idx in Nearest_Points[1]:
        point = SB_Sim_Points[idx]
        distances, neighbors = Sim_Tree.query(point, k=nearest_neighbors)
        
        # Collect data for each neighbor
        for neighbor_idx in neighbors:
            neighbor_point = SB_Sim_Points[neighbor_idx]
            angle = Sim_dexes.angle_calc(point, neighbor_point)
            distance = Sim_dexes.dist_calc(point, neighbor_point)
            points.append((idx, neighbor_idx, distance, angle))

    # Create a dictionary to group points
    points_dict = {}
    num_points = len(points)

    for group_index in range(1, min(nearest_neighbors**2 // nearest_neighbors + 1, num_points // nearest_neighbors + 1)):
        start_index = (group_index - 1) * nearest_neighbors
        # Ensure that we have enough points to make a full group
        if start_index + 4 < num_points:
            points_dict[group_index] = points[start_index:start_index + 4]

    print(points_dict)

    # Initialize dictionaries to store matches at different levels
    matches = {}
    matches2 = {}
    matches3 = {}
    matches4 = {}
    multiplier = 2

    # Helper function to find matches based on error tolerance
    def find_matches(current_matches, i, lookup_table, error_percent, error_percent_angle, multiplier):
        new_matches = []
        for found in current_matches:
            for key, val in lookup_table.items():
                for item in val:
                    if found == item[0]:
                        if abs((i[2] - key[0]) / key[0]) <= (error_percent * multiplier):
                            if abs((i[3] - key[1]) / key[1]) <= (error_percent_angle * multiplier):
                                new_matches.append(item[0])
        print(f'newmatches{new_matches}')
        return new_matches

    aa = 0
    # Iterate through points_dict items
    for a, b in points_dict.items():
        # print(f'a: {a} b: {b}')

        # Step 1: Find initial matches
        initial_matches = []
        for point in b:
            print(f'point{point}')
            if point[2] != 0:
                for key, val in lookup_table.items():
                    for item in val:
                        if key[0] != 0:
                            if abs((point[2] - key[0]) / key[0]) <= (error_percent * multiplier):
                                if key[1] != 0:
                                    if abs((point[3] - key[1]) / key[1]) <= (error_percent_angle * multiplier):
                                        initial_matches.append(item[0])
        matches[a] = initial_matches
        print(f'init_matches{matches}')

        aa = aa + 1

        # Step 2-4: Find subsequent matches based on previous results
        current_matches = matches
        for count in range(aa-1):
            if len(current_matches) == 0:  # No matches found in the previous step
                print('NO INITIAL MATCHES')
                break

            # Adjust the multiplier for each step
            current_multiplier = multiplier if count == 1 else multiplier * 2
            
            # Find matches for the current step
            next_matches = find_matches(current_matches, b[count], lookup_table, error_percent, error_percent_angle, current_multiplier)
            if len(next_matches) > 0:
                if count == 1:
                    matches2[a] = next_matches
                    current_matches = matches2[a]
                elif count == 2:
                    matches3[a] = next_matches
                    current_matches = matches3[a]
                elif count == 3:
                    matches4[a] = next_matches
                    current_matches = matches4[a]

    print(matches4)
    print(max(lat), max(lon))
    print(min(lat), min(lon))

    lat1 = []
    lon1 = []

    print(matches4.items())

    # Determine the best matches from matches4 down to matches
    if matches4:
        source_matches = matches4
        print('great matches')
    elif matches3:
        source_matches = matches3
        print('good matches')
    elif matches2:
        source_matches = matches2
        print('okay matches')
    elif matches:
        source_matches = matches
        print('bad matches')
    else:
        source_matches = None
        print('no matches')

    # Populate lat1 and lon1 based on the best available matches
    if source_matches:
        for key, bingo in source_matches.items():
            if len(bingo) > 0:
                max_index = max_idx(bingo)
                print(f'{key}: {max_index}')
                lat1.append(lat[max_index])
                lon1.append(lon[max_index])

    lat2 = []
    lon2 = []
    for i in Nearest_Points_WF[1]:
        print(i)
        # print(lat[i])
        # print(lon[i])
        lat2.append(lat[i])
        lon2.append(lon[i])

    print(lat2)
    print(lon2)
    #Use to confirm your (pseudo) matches are actually being calculated correctly
    Match_points_lat = SB_Point_lat + Match_points_lat
    Match_points_lon = SB_Point_lon + Match_points_lon

    plt.figure(figsize=(10, 6))
    plot_coordinates(lat, lon, agl, SB_Point_lat, SB_Point_lon, Match_points_lat, Match_points_lon, lat1, lon1, lat2, lon2, colour='blue')
    # plot_coordinates(SB_Point_lat, SB_Point_lon, 1, colour='green')
    plt.title('Latitude and Longitude Plot')
    plt.legend()
    plt.show()

    hi = 'hi'
    print(hi)
    # jnsdlcnl

if __name__ == "__main__":
    
    filename = './DOF/38-ND.Dat'
    filename1 = './DOF/27-MN.Dat'
    filename2 = './DOF/18-IN.Dat'
    filename3 = './DOF/50-VT.Dat'
    filename4 = './DOF/27-MN.Dat'
    filename5 = './DOF/30-MT.Dat'
    filename6 = './DOF/46-SD.Dat'
    
    lat, lon, agl, msl = [], [], [], []
    lat, lon, agl, msl = read_lat_lon(filename, lat, lon, agl, msl, MIN_AGL=50)
    min_lat_nd, max_lat_nd = min(lat), max(lat)
    min_lon_nd, max_lon_nd = min(lon), max(lon)
    lat, lon, agl, msl = read_lat_lon(filename4, lat, lon, agl, msl, MIN_AGL=50)
    lat, lon, agl, msl = read_lat_lon(filename5, lat, lon, agl, msl, MIN_AGL=50)
    lat, lon, agl, msl = read_lat_lon(filename6, lat, lon, agl, msl, MIN_AGL=50)
    

    size = 5
    
    # SB_Point_lat = 47.83184
    # SB_Point_lon = -97.0043
    # SB_Point = [SB_Point_lat, SB_Point_lon]
    # print(f'Drone location:', SB_Point)

    random_points = generate_random_points_in_polygon(lat, lon, size)

    LofS = 0.2 ## visibility in degrees
    
    indexes = KD_calcs('distances')
    indexes, index_a, index_b, distance, angle, lat_min, lat_max, lon_min, lon_max, agl_min, agl_max, Tree, KDarray = KD_calcs.create_KD_Tree(indexes, lat, lon, agl)

    lookup_table = KD_calcs.Lookup_Table(indexes, index_a, index_b, distance, angle)
    
    # for i in random_points:
    #     SB_Point = i
    #     SB_Point_lat = i[0]
    #     SB_Point_lon = i[1]
    #     main(indexes, index_a, index_b, distance, angle, lat_min, lat_max, lon_min, lon_max, agl_min, agl_max, Tree, KDarray, lookup_table, LofS)

    # counts, less_than = random_point_within_radius(random_points, Tree, LofS)
    # plot_histogram(counts)
    # plot_coordinates_2(less_than)

    DOF_heatmap(lat, lon)

    hi = 'hi'
    print(hi)
