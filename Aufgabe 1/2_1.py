#!/usr/bin/env python3
# Python 3.7

import sys
import os
import math
import matplotlib.pyplot as plt
import shapely.geometry as geometry
import networkx as nx
import datetime

BUS_SPEED = 30 / 3.6
RUN_SPEED = 15 / 3.6


def main():
    # Festlegen der Standarddatei
    file_name = os.path.join(os.path.dirname(__file__), "lisarennt3.txt")

    if len(sys.argv) == 2:
        file_name = os.path.join(os.path.dirname(__file__), sys.argv[1])

    # Einlesen der Datei
    input_file = open(file_name, 'r')
    str_data = input_file.read().splitlines()
    input_file.close()

    str_data_split = [list(map(int, string.split(' '))) for string in str_data]

    poly_data = str_data_split[1:len(str_data_split) - 1]
    raw_polygons = []
    all_points = []

    # Erstellen der Polygone
    for data_line in poly_data:
        vertices = []
        for _ in range(1, len(data_line) - 1, 2):
            vertices.append((data_line[_], data_line[_ + 1]))
        raw_polygons.append(Polygon(vertices))
        all_points += vertices

    start_point = (str_data_split[-1][0], str_data_split[-1][1])
    all_points.append(start_point)

    # Vereinen von Polygonen, die einander beruehren
    polygons = []
    for poly1 in raw_polygons:
        for poly2 in raw_polygons:
            if poly1 is not poly2 and poly1.poly_obj.intersects(poly2.poly_obj):
                    poly1 = poly1.poly_obj.union(poly2.poly_obj)
                    # Aktualisieren der Liste der Ecken des neuen kombinierten Polygons
                    poly1.vertices = [(round(p[0]), round(p[1])) for p in
                                      geometry.mapping(poly1.poly_obj)['coordinates'][0]]
                    raw_polygons.remove(poly2)
        polygons.append(poly1)

    # Erstellen eines Graphen mit allen bisherigen Punkten
    graph = nx.Graph()
    graph.add_nodes_from(all_points)

    # Erstellen von erlaubten Verbindungen innerhalb des Graphen
    checked_connections = []
    for polygon in polygons:
        for p1 in polygon.vertices:
            for p2 in all_points:
                if p1 != p2 and (p2, p1) not in checked_connections:
                    valid_connection, connecting_line = check_edge_valid(p1, p2, polygons, polygon)
                    if valid_connection:
                        graph.add_edge(p1, p2, weight=connecting_line.length)
                    checked_connections.append((p1, p2))

    # Hinzufuegen moeglicher Endpunkte an der Gerade x = 0
    destinations = []
    for p1 in all_points:
        p2 = calculate_ideal_finish(p1)
        valid_connection, connecting_line = check_edge_valid(p1, p2, polygons)
        if valid_connection:
            graph.add_node(p2)
            graph.add_edge(p1, p2, weight=connecting_line.length)
            destinations.append((p2, nx.astar_path_length(graph, start_point, p2,
                                                          lambda p, q: geometry.LineString([p, q]).length)))

    # Ausrechnen der Guete jedes Endpunktes
    early_start = []
    for point, distance in destinations:
        early_start.append(distance / RUN_SPEED - point[1] / BUS_SPEED)

    best_destination = destinations[early_start.index(min(early_start))]
    best_path = nx.astar_path(graph, start_point, best_destination[0], lambda p, q: geometry.LineString([p, q]).length)

    # Ausgabe von Routeninformationen
    start_time = (datetime.datetime(year=1, month=1, day=1, hour=7, minute=30)
                  - datetime.timedelta(seconds=min(early_start)))
    print("Laufdistanz: " + str(best_destination[1]) + " m")
    print("Laufzeit: " + str(best_destination[1] / RUN_SPEED) + " s")
    print("Startzeit: " + start_time.time().strftime('%H:%M:%S.%f'))
    print("Ankunftszeit: " + (start_time + datetime.timedelta(seconds=best_destination[1] / RUN_SPEED))
          .time().strftime('%H:%M:%S.%f'))
    print("Treffpunkt mit Bus: " + str(best_destination[0]))
    print()
    print("Route:")
    # Ausgabe der Route, Feststellen der jeweiligen Polygon-ID
    for i in range(len(best_path)):
        point = best_path[i]
        point_tag = ""
        if i == 0:
            point_tag = "L"
        elif i == (len(best_path) - 1):
            point_tag = "Bus"
        else:
            for j in range(len(raw_polygons)):
                if raw_polygons[j].poly_obj.intersects(geometry.Point(point)):
                    point_tag = "P" + str(j + 1)
                    break
        print(str(i + 1) + ". " + str((point[0], point[1], point_tag)))

    # Zeichnen des Weges
    plot_polygons(polygons)
    plot_line(best_path, 'black')
    plt.axvline(x=0, color='grey')
    plt.scatter(best_path[0][0], best_path[0][1], color='green', label='Start')
    plt.scatter(best_path[-1][0], best_path[-1][1], color='red', label='Ende')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Klasse, um Informationen ueber Polygone zu speichern
class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices
        self.poly_obj = geometry.Polygon(vertices)


# Prueft, ob eine verbindende Kante zwischen zwei Punkten keine anderen Polygone schneidet
def check_edge_valid(p1, p2, polygons, polygon=None):
    connecting_line = geometry.LineString([p1, p2])
    intersecting_geometry = []
    for polygon2 in polygons:
        current_intersection = polygon2.poly_obj.intersection(connecting_line)
        if isinstance(current_intersection, geometry.Point) or \
                isinstance(current_intersection, geometry.LineString):
            intersecting_geometry.append(current_intersection)
        else:
            intersecting_geometry += list(current_intersection)
    valid_connection = True
    for geom in intersecting_geometry:
        validated = False
        if polygon is not None:
            if polygon.poly_obj.boundary.contains(geom):
                validated = True
        else:
            if geometry.Point(p1).contains(geom):
                validated = True
        if geometry.Point(p2).contains(geom):
            validated = True
        if not validated:
            valid_connection = False
    return valid_connection, connecting_line


# Zeichnet eine Linie im pyplot-Diagramm
def plot_line(vertices, color='#333333'):
    x = []
    y = []
    for vertex in vertices:
        x.append(vertex[0])
        y.append(vertex[1])
    plt.plot(x, y, color)


# Zeichnet ein Polygon
def plot_polygons(polygons):
    for polygon in polygons:
        x = []
        y = []
        for vertex in polygon.vertices:
            x.append(vertex[0])
            y.append(vertex[1])
        plt.fill(x, y)


# Berechnet die ideale ziel-y-Koordinate abhaengig vom aktuellen Standort
def calculate_ideal_finish(coords):
    return 0, coords[1] + ((RUN_SPEED * coords[0]) / math.sqrt(math.pow(BUS_SPEED, 2) - math.pow(RUN_SPEED, 2)))


if __name__ == '__main__':
    main()
