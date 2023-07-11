import gym
import pixelate_arena
import time
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import math
from collections import deque, namedtuple
import cv2.aruco as aruco
import timeit

inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def goto(current_point, point):
    i = 0
    target_dir = math.atan2((point[1] - current_point[1]), (point[0] - current_point[0]))
    # print(husky_dir, target_dir)
    current_point = (
        tiles_grid[current_point[0]][current_point[1]][0], tiles_grid[current_point[0]][current_point[1]][1])
    point = (tiles_grid[point[0]][point[1]][0], tiles_grid[point[0]][point[1]][1])
    # print(current_point)
    while True:
        if i % 15 == 0:
            image = env.camera_feed()
            image = np.ascontiguousarray(image, dtype=np.uint8)
            cv2.imshow("img", image)
            cv2.imwrite("sample_arena_img.png", image)
            cv2.waitKey(1)
            corners = get_corners(image)
            if corners is not None:
                current_point = ((corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2)
                # print(current_point)
                husky_dir = math.atan2((corners[0][1] - corners[3][1]), (corners[0][0] - corners[3][0]))
                target_dir = math.atan2((point[1] - current_point[1]), (point[0] - current_point[0]))
        # print(husky_dir, target_dir)
        i += 1
        p.stepSimulation()

        velocity = 7
        if math.fabs(husky_dir - target_dir) < 0.05:
            env.move_husky(0, 0, 0, 0)
            break
        else:
            if target_dir * husky_dir > 0:
                velocity = (target_dir - husky_dir) * 15
                env.move_husky(velocity, -velocity, velocity, -velocity)
            else:
                if husky_dir < 0:
                    if 0 <= target_dir - husky_dir <= math.pi:
                        env.move_husky(velocity, -velocity, velocity, -velocity)
                    else:
                        env.move_husky(-velocity, velocity, -velocity, velocity)
                else:
                    if 0 >= target_dir - husky_dir >= -math.pi:
                        env.move_husky(-velocity, velocity, -velocity, velocity)
                    else:
                        env.move_husky(velocity, -velocity, velocity, -velocity)

    i = 0
    while True:
        if i % 15 == 0:
            image = env.camera_feed()
            image = np.ascontiguousarray(image, dtype=np.uint8)
            p.stepSimulation()
            cv2.imshow("img", image)
            cv2.imwrite("sample_arena_img.png", image)
            cv2.waitKey(1)
            corners = get_corners(image)
            if corners is not None:
                current_point = ((corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2)
                # print(point, current_point)
                distance = math.sqrt((point[1] - current_point[1]) ** 2 + (point[0] - current_point[0]) ** 2)
                # print(distance)
        i += 1
        p.stepSimulation()
        if distance < 7:
            env.move_husky(0, 0, 0, 0)
            break
        else:
            if distance > 12:
                env.move_husky(12, 12, 12, 12)
            else:
                env.move_husky(distance/2, distance/2, distance/2, distance/2)
            # env.move_husky(8, 8, 8, 8)


def get_corners(img):
    # Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
        markersX=2,
        markersY=2,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

    # img = cv2.imread('sample_arena_img.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Make sure all 5 markers were detected before printing them out
    # if ids is not None:
    #     # Print corners and ids to the console
    #     for i, corner in zip(ids, corners):
    #         print('ID: {}; Corners: {}'.format(i, corner))
    # Outline all of the markers detected in our image
    # print(corners)
    if corners is not None:
        if corners[0] is not None:
            if len(corners[0][0]):
                img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))
                img = cv2.rectangle(img, (int(corners[0][0][0][0]), int(corners[0][0][0][1])),
                                    (int(corners[0][0][2][0]), int(corners[0][0][2][1])), (0, 255, 0), 2)

                return corners[0][0]
            else:
                return None
        else:
            return None
    else:
        return None


def make_edge(start, end, cost=1):
    return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(

            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'

        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:

            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])

            if distances[current_vertex] == inf:
                break

            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost

                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

            vertices.remove(current_vertex)

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path, distances[dest]


if __name__ == "__main__":
    env = gym.make("pixelate_arena-v0")
    x = 0
    env.remove_car()
    p.stepSimulation()
    img = env.camera_feed()
    img = np.ascontiguousarray(img, dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # COLOR_MASKS
    white_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([0, 255, 255]))
    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([60, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([30, 255, 255]))
    purple_mask = cv2.inRange(hsv, np.array([140, 0, 0]), np.array([150, 255, 255]))
    pink_mask = cv2.inRange(hsv, np.array([160, 0, 0]), np.array([170, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([60, 0, 0]), np.array([140, 255, 255]))

    # COLOR_CONTOURS
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, white_contours, -1, (255, 0, 0), 3)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, red_contours, -1, (255, 0, 0), 3)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, green_contours, -1, (255, 0, 0), 3)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, yellow_contours, -1, (255, 0, 0), 3)
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, purple_contours, -1, (255, 0, 0), 3)
    pink_contours, _ = cv2.findContours(pink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, pink_contours, -1, (255, 0, 0), 3)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, blue_contours, -1, (255, 0, 0), 3)

    # LIST_OF_COLOR_CONTOURS
    color_contours = [white_contours, red_contours, green_contours, yellow_contours, purple_contours, pink_contours, blue_contours]
    damage_list = [1, 1, 4, 2, 3, 1, 200]
    colors_list = []
    for contours in color_contours:
        color_list = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                cx = int(M['m10'] / (M['m00'] + 0.0001))
                cy = int(M['m01'] / (M['m00'] + 0.0001))
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
            color_list.append((cx, cy))
        colors_list.append(color_list)

    all_tiles = colors_list[0] + colors_list[1] + colors_list[2] + colors_list[3] + colors_list[4] + colors_list[5] + colors_list[6]
    all_tiles = list(set(all_tiles))
    all_tiles = sorted(all_tiles, key=lambda x: x[1])
    # print(all_tiles)
    tiles_grid = []
    start = 0
    for i in range(len(all_tiles) - 1):
        if all_tiles[i + 1][1] - all_tiles[i][1] > 3:
            tiles_grid.append(all_tiles[start:i + 1])
            start = i + 1
        elif i + 1 == len(all_tiles) - 1:
            tiles_grid.append(all_tiles[start:])

    # print(tiles_grid)
    for i in range(len(tiles_grid)):
        tiles_grid[i] = sorted(tiles_grid[i], key=lambda x: x[0])
        print(tiles_grid[i])
        j=0
        while j < len(tiles_grid[i]) - 1:
            print(tiles_grid[i][j][0])
            print(tiles_grid[i][j + 1])
            if math.fabs(tiles_grid[i][j][0] - tiles_grid[i][j + 1][0]) < 2:
                tiles_grid[i].remove(tiles_grid[i][j])
            else:
                j += 1

    # print(tiles_grid)
    for i in range(len(tiles_grid)):
        j = 0
        while j < len(tiles_grid[i]) - 1:
            if tiles_grid[i][j] != (-1, -1):
                count = (tiles_grid[i][j + 1][0] - tiles_grid[i][j][0]) // 44 - 1
                for k in range(count):
                    tiles_grid[i].insert(j + 1, (-1, -1))
            j += 1
        # print(tiles_grid[i])
    # print()
    for i in range(13):
        for j in range(int(math.fabs(6 - i))):
            tiles_grid[i].insert(0, (-1, -1))
        if i <= 6:
            count = 0
            for j in range(6 + i):
                tiles_grid[i].insert(7 - i + j + count, (-1, -1))
                count += 1
        else:
            count = 0
            for j in range(18 - i):
                tiles_grid[i].insert(i + j + count - 5, (-1, -1))
                count += 1
        for j in range(int(math.fabs(6 - i))):
            tiles_grid[i].append((-1, -1))
    # print()
    # for x in tiles_grid:
    #     print(x)

    node_list = []
    for i in range(13):
        for j in range(25):
            if tiles_grid[i][j] != (-1, -1):
                neighbour_nodes = [(i, j - 2), (i, j + 2), (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1),
                                   (i + 1, j + 1)]
                for node in neighbour_nodes:
                    try:
                        if tiles_grid[node[0]][node[1]] != (-1, -1):
                            if tiles_grid[node[0]][node[1]] in colors_list[6]:
                                node_list.append(((i, j), node, damage_list[6]))
                            else:
                                for k in range(6):
                                    if tiles_grid[node[0]][node[1]] in colors_list[k]:
                                        node_list.append(((i, j), node, damage_list[k]))
                    except IndexError:
                        pass

    # print(node_list)
    graph = Graph(node_list)
    env.respawn_car()
    p.stepSimulation()
    img = env.camera_feed()
    img = np.ascontiguousarray(img, dtype=np.uint8)
    cv2.imshow("img", img)
    cv2.imwrite("sample_arena_img.png", img)
    cv2.waitKey(1)
    corners = get_corners(img)
    current_pos = ((corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2)

    spidey_loc = []
    s1_loc = ()
    for point in colors_list[1]:
        for i in range(13):
            for j in range(25):
                if tiles_grid[i][j] == point:
                    spidey_loc.append((i, j))
                    if math.fabs(point[0] - current_pos[0]) < 10 and math.fabs(point[0] - current_pos[0]) < 10:
                        s1_loc = (i, j)

    path1, distance1 = [], np.inf
    for loc in spidey_loc:
        if loc != s1_loc:
            path, distance = graph.dijkstra(s1_loc, loc)
            if distance < distance1:
                path1, distance1 = list(path), distance

    path2 = []
    for loc in spidey_loc:
        if loc != path1[0] and loc != path1[len(path1) - 1]:
            path2, distance2 = graph.dijkstra(path1[len(path1) - 1], loc)
            path2 = list(path2)

    curr_point = path1[0]
    for i in range(1, len(path1)):
        goto(curr_point, path1[i])
        curr_point = path1[i]
    print("\nREACHED TO SPIDER-MAN 2")

    flag = 0
    for i in range(1, len(path2)):
        goto(curr_point, path2[i])
        curr_point = path2[i]
        flag = 1
    if flag:
        print("REACHED TO SPIDER-MAN 3")

    # ***** DETECTING ANTIDOTES ***** #

    print("PLATES REMOVED OVER THE ANTIDOTES")
    env.unlock_antidotes()
    antidotes = {}
    villains = {}

    p.stepSimulation()
    img = env.camera_feed()
    img = np.ascontiguousarray(img, dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([60, 0, 0]), np.array([140, 255, 255]))

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, blue_contours, -1, (0, 0, 255), 2)
    tcarea = []
    tcpoints = []
    for contour in blue_contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M['m00'] == 0:
            cx = int(M['m10'] / (M['m00'] + 0.0001))
            cy = int(M['m01'] / (M['m00'] + 0.0001))
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
        # print(area, contour)
        for point in colors_list[4]:
            if math.fabs(point[0] - cx) < 5 and math.fabs(point[1] - cy) < 5:
                villains["T"] = point

        for point in colors_list[3]:
            if math.fabs(point[0] - cx) < 5 and math.fabs(point[1] - cy) < 5:
                villains["C"] = point

        for point in colors_list[2]:
            if math.fabs(point[0] - cx) < 5 and math.fabs(point[1] - cy) < 5:
                villains["S"] = point

        for point in colors_list[5]:
            if math.fabs(point[0] - cx) < 5 and math.fabs(point[1] - cy) < 5:
                if len(contour) == 4:
                    antidotes["S"] = point
                else:
                    tcarea.append(area)
                    tcpoints.append(point)

    if tcarea[0] < tcarea[1]:
        antidotes["T"] = tcpoints[0]
        antidotes["C"] = tcpoints[1]
    else:
        antidotes["C"] = tcpoints[0]
        antidotes["T"] = tcpoints[1]

    for i in range(13):
        for j in range(25):
            for shape in villains:
                if math.fabs(tiles_grid[i][j][0] - villains[shape][0]) < 5 and math.fabs(
                        tiles_grid[i][j][1] - villains[shape][1]) < 5:
                    villains[shape] = (i, j)
            for shape in antidotes:
                if math.fabs(tiles_grid[i][j][0] - antidotes[shape][0]) < 5 and math.fabs(
                        tiles_grid[i][j][1] - antidotes[shape][1]) < 5:
                    antidotes[shape] = (i, j)
    # print(villains)
    # print(antidotes)
    if len(villains) > 1 and len(antidotes) > 1:
        possible_paths = ["CTSCTS", "CSTCST", "TCSTCS", "TSCTSC", "SCTSCT", "STCSTC"]

        total_distance = np.inf
        final_path = ""
        final_path_list = []
        for pathx in possible_paths:
            t_distance = 0
            temp_path_list = []
            # print(pathx)
            for i in range(6):
                if i < 3:
                    if i == 0:
                        path, distance = graph.dijkstra(path2[len(path2)-1], antidotes[pathx[i]])
                        t_distance += distance
                        temp_path_list.append(list(path))
                    else:
                        path, distance = graph.dijkstra(antidotes[pathx[i-1]], antidotes[pathx[i]])
                        t_distance += distance
                        temp_path_list.append(list(path))
                elif i == 3:
                    path, distance = graph.dijkstra(antidotes[pathx[2]], villains[pathx[i]])
                    t_distance += distance
                    temp_path_list.append(list(path))
                else:
                    path, distance = graph.dijkstra(villains[pathx[i-1]], villains[pathx[i]])
                    t_distance += distance
                    temp_path_list.append(list(path))
            # print(t_distance)
            if t_distance < total_distance:
                total_distance = t_distance
                final_path = pathx
                final_path_list = temp_path_list

        # print(final_path, total_distance)
        indication = []
        for i in range(3):
            if final_path[i] == "T":
                indication.append("Antidote for Sandman Collected")
            if final_path[i] == "C":
                indication.append("Antidote for Electro Collected")
            if final_path[i] == "S":
                indication.append("Antidote for Goblin Collected")
        for i in range(3, 6):
            if final_path[i] == "T":
                indication.append("Sandman Cured")
            if final_path[i] == "C":
                indication.append("Electro Cured")
            if final_path[i] == "S":
                indication.append("Goblin Cured")
        print(indication)
        j = 0
        for path in final_path_list:
            curr_point = path[0]
            for i in range(1, len(path)):
                goto(curr_point, path[i])
                curr_point = path[i]
            # print(j)
            print(indication[j])
            j += 1
            if j == 3:
                print("ALL ANTIDOTES COLLECTED")
        print("ALL VILLAINS CURED\n")
    else:
        antidote_pos = [value for key, value in antidotes.items()]
        pathA, _ = graph.dijkstra(path1[len(path1)-1], antidote_pos[0])
        villain_pos = [value for key, value in villains.items()]
        pathB, _ = graph.dijkstra(antidote_pos[0], villain_pos[0])
        curr_point = pathA[0]
        for i in range(1, len(pathA)):
            goto(curr_point, pathA[i])
            curr_point = pathA[i]
        print("ANTIDOTE COLLECTED")
        for i in range(1, len(pathB)):
            goto(curr_point, pathB[i])
            curr_point = pathB[i]
        print("VILLAIN CURED\n")

    # time.sleep(1)

