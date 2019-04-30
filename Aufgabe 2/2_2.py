#!/usr/bin/env python3
# Python 3.7

import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import itertools


def main():
    # Festlegen der Standarddatei
    file_name = os.path.join(os.path.dirname(__file__), "dreiecke3.txt")

    if len(sys.argv) == 2:
        file_name = os.path.join(os.path.dirname(__file__), sys.argv[1])

    # Einlesen der Datei
    input_file = open(file_name, 'r')
    str_data = input_file.read().splitlines()
    input_file.close()
    str_data_split = [list(map(int, string.split(' '))) for string in str_data]

    # Erzeugen der Dreiecke
    triangles = [Triangle([np.array([element[i], element[i + 1]]) for i in range(1, 6, 2)], 'D' + str(index + 1))
                 for index, element in enumerate(str_data_split[1:])]

    # Optimieren der Parameter a, b und c
    epochs = 30 if len(triangles) < 50 else round(30 * 50 / (len(triangles)**1.4))
    a, b, c = optimize(triangles, epochs)

    # Anordnung der Dreiecke
    arranged_triangles = arrange(triangles, delta_a1=a, delta_a2=b, delta_b=c)

    # Ausgabe von Text und Grafik
    for triangle in arranged_triangles:  # TODO remove splice
        triangle.print()
        triangle.plot()
    print('Gesamtdistanz: ' + str(round(arranged_triangles[-1].x, 3)))
    plt.axis('equal')
    plt.axhline(y=0, color='grey')
    plt.show()


# Findet eine moegliche Anordnung der Dreiecke
def arrange(p_triangles, delta_a1=10, delta_a2=25, delta_b=30):
    triangles = copy.deepcopy(p_triangles)
    current_x = 0
    current_angle = 0
    placed_triangles = []

    while triangles:
        remaining_angle = 180 - current_angle
        # Erstellt eine Liste aller Winkel und der Indizes ihres zugehoerigen Dreiecks
        angles = list(itertools.chain(*[[(index, angle) for angle in triangle.angles]
                                        for index, triangle in enumerate(triangles)]))

        # Sucht ein Dreieck, das gut in den Winkel passt, der aktuell offen ist
        triangle = None
        for index, angle in angles:
            if abs(remaining_angle - angle) < delta_a1 and angle < delta_a2:
                triangle = triangles[index]
                break

        # Wird ausgefuehrt, wenn ein potentielles Dreieck gefunden wurde
        if triangle is not None:
            current_x, current_angle = triangle.place_on_fitting_angle(current_x, current_angle, placed_triangles)
        else:
            # Falls kein Dreieck die Luecke gut fuellt, wird stattdessen ein Dreieck im aktuellen Punkt plaziert
            min_angles = [(index, min(triangle.angles)) for index, triangle in enumerate(triangles)]
            triangle = None
            for index, angle in min_angles:
                if angle + delta_b < remaining_angle:
                    triangle = triangles[index]
                    break

            if triangle is not None:
                current_x, current_angle = triangle.place_on_fitting_side(current_x, current_angle, placed_triangles)
            # Passt kein Dreieck in den aktuellen Punkt,
            # wird das am besten passende Dreieck entlang der x-Achse plaziert
            else:
                angle_diffs = [(index, abs(remaining_angle - angle)) for index, angle in angles]
                triangle = triangles[min(angle_diffs, key=lambda n: n[1])[0]]
                current_x, current_angle = triangle.place_on_fitting_angle(current_x, current_angle, placed_triangles)

        placed_triangles.append(triangle)
        remove_by_id(triangles, triangle)

    return placed_triangles


# Optimiert die Parameter a, b, c fuer die gegebenen Daten
def optimize(data, epochs):
    a, b, c = 10, 25, 30

    arrangement = arrange(data, delta_a1=a, delta_a2=b, delta_b=c)
    baseline = arrangement[-1].x
    parameters = {baseline: (a, b, c)}
    # repetitions = 0

    for i in range(epochs):
        print('\r' + '#' * (i + 1) + '-' * (epochs - (i + 1)) + ' '
              + str(i + 1) + '/' + str(epochs), flush=True, end='')
        delta = 10 / (1.02 ** i)

        a += 0.1 * delta
        arrangement = arrange(data, delta_a1=a, delta_a2=b, delta_b=c)
        a_x = arrangement[-1].x
        if a_x > baseline:
            a -= 2 * 0.1 * delta
        elif math.isclose(a_x, baseline):
            a -= 0.1 * delta
        parameters[a_x] = (a, b, c)

        b += delta
        arrangement = arrange(data, delta_a1=a, delta_a2=b, delta_b=c)
        b_x = arrangement[-1].x
        if b_x > baseline:
            b -= 2 * delta
        elif math.isclose(b_x, baseline):
            b -= delta
        parameters[b_x] = (a, b, c)

        c += delta
        arrangement = arrange(data, delta_a1=a, delta_a2=b, delta_b=c)
        c_x = arrangement[-1].x
        if c_x > baseline:
            c -= 2 * delta
        elif math.isclose(c_x, baseline):
            c -= delta
        parameters[c_x] = (a, b, c)
        baseline = c_x

    print('\r ')
    return parameters[min(parameters)]


class Triangle:
    def __init__(self, vertices, identification=''):
        self.angles = [calculate_angle(vertices[i - 1], vertices[i], vertices[(i + 1) % 3]) for i in range(3)]
        self.lengths = [np.linalg.norm(vertices[i - 1] - vertices[(i + 1) % 3]) for i in range(3)]
        self.x = None
        self.rotation_angle = None
        self.base_angle = None
        self.left_side = None
        self.right_side = None
        self.placement_type = None
        self.ID = identification

    # Plaziert das Dreieck so, dass der Kleinste Winkel an der x-Achse liegt
    def place_on_fitting_side(self, x, angle, placed_triangles):
        self.placement_type = 'side'
        self.x = x
        self.rotation_angle = angle

        # Der kleinste Winkel soll als Basis dienen
        smallest_angle_index = self.angles.index(min(self.angles))
        self.base_angle = self.angles[smallest_angle_index]

        # Wenn der aktuelle Platzierungswinkel kleiner als 90 Grad ist, soll die groessere Kante rechts liegen,
        # wenn er groesser als 90 Grad ist, links, ausser, die x-Position ist 0
        l1 = self.lengths[smallest_angle_index - 1]
        l2 = self.lengths[(smallest_angle_index + 1) % 3]
        longer = l1 if l1 > l2 else l2
        shorter = l2 if l1 > l2 else l1
        self.left_side = longer if angle > 90 or x == 0 else shorter
        self.right_side = shorter if angle > 90 or x == 0 else longer

        # Justieren des Winkels, damit keine Kollision mit vorher platzierten Dreiecken stattfindet
        for triangle in placed_triangles:
            if triangle.placement_type == 'angle':
                p1 = triangle.get_right_top_vertex()
                p2 = triangle.get_left_top_vertex()
                p3 = triangle.get_base_vertex()
            else:
                p1 = triangle.get_base_vertex()
                p2 = triangle.get_right_top_vertex()
                p3 = triangle.get_left_top_vertex()

            qb, ql, qr = self.get_vertices()
            if intersects(p1, p2, qb, ql):
                self.adjust_angle(p1, p2, True)

            qb, ql, qr = self.get_vertices()
            if intersects(p3, p1, qb, ql) or intersects(p3, p1, ql, qr):
                self.adjust_angle(p2, p3, True)

            qb, ql, qr = self.get_vertices()
            if intersects(p1, p2, qr, qb):
                self.adjust_angle(p1, p2, False)

            qb, ql, qr = self.get_vertices()
            if intersects(p1, p2, qb, ql) or intersects(p2, p3, qb, ql):
                self.adjust_angle(p2, p3, True)

            qb, ql, qr = self.get_vertices()
            if intersects(p2, p3, qr, qb):
                self.adjust_angle(p2, p3, False)

            qb, ql, qr = self.get_vertices()
            if intersects(p2, p3, ql, qr):
                self.adjust_angle(p2, p3, True, choose_max=True)

        new_x = x
        new_angle = self.rotation_angle + self.base_angle
        return new_x, new_angle

    # Justiert den Winkel so, dass keine Kollision mit der Kante p1p2 vorliegt
    def adjust_angle(self, p1, p2, left, choose_max=False):
        beta_rad = np.arctan((p1[1] - p2[1]) / (p1[0] - p2[0]))
        c = self.x - p1[0]
        b = self.left_side if left else self.right_side
        max_angle = np.rad2deg(np.arctan(p2[1] / (self.x - p2[0])))
        max_angle = max_angle if max_angle >= 0 else max_angle + 180
        if not choose_max:
            gamma_rad = np.arcsin(c * np.sin(beta_rad) / b)
            alpha = np.rad2deg(np.pi - beta_rad - gamma_rad)
            self.rotation_angle = max_angle if alpha > max_angle else alpha
        else:
            self.rotation_angle = max_angle

    # Plaziert das Dreieck so, dass die am besten passende Ecke den restlichen Winkel fuellt,
    # und eine Kante an der x-Achse liegt
    def place_on_fitting_angle(self, x, angle, placed_triangles):
        self.placement_type = 'angle'
        remaining_angle = 180 - angle
        # Findet den Winkel mit der kleinsten positiven Differenz zum gesuchten Wert
        best_angle_index = min(enumerate([remaining_angle - current_angle for current_angle in self.angles]),
                               key=lambda n: n[1] if n[1] >= 0 else (abs(n[1] - 180)))[0]

        self.base_angle = self.angles[best_angle_index]
        self.x = x

        l1 = self.lengths[best_angle_index - 1]
        l2 = self.lengths[(best_angle_index + 1) % 3]
        self.left_side = l2 if l1 > l2 else l1
        self.right_side = l1 if l1 > l2 else l2
        self.rotation_angle = 180 - self.base_angle  # Das Dreieck soll flach auf dem Boden liegen

        # Verschieben des Dreiecks nach Rechts, falls es nicht in die aktuelle Luecke passt
        if len(placed_triangles) != 0 and self.base_angle > remaining_angle:
            self.adjust_x(placed_triangles[-1])

        for triangle in placed_triangles:
            p1, p3, p2 = triangle.get_vertices()
            qb, ql, qr = self.get_vertices()
            if intersects(p1, p2, qb, ql) or intersects(p2, p3, qb, ql) or intersects(p3, p1, qb, ql) \
                    or intersects(p1, p2, ql, qr) or intersects(p2, p3, ql, qr) or intersects(p3, p1, ql, qr):
                self.adjust_x(triangle)

        new_x = self.x + self.right_side
        new_angle = calculate_angle(np.array([self.x, 0]), self.get_right_top_vertex(), self.get_left_top_vertex())
        return new_x, new_angle

    # Passt den x-Wert des Dreiecks so an, dass es in die Luecke passt
    def adjust_x(self, triangle):
        if triangle.get_right_top_vertex()[1] <= self.get_left_top_vertex()[1]:
            self.x = triangle.get_right_top_vertex()[0] \
                     - triangle.get_right_top_vertex()[1] / np.tan(np.deg2rad(self.base_angle))
        else:
            y_diff = triangle.get_right_top_vertex()[1] - self.get_left_top_vertex()[1]
            m_triangle = np.tan(np.deg2rad(180 - triangle.rotation_angle - triangle.base_angle))
            m_self = np.tan(np.deg2rad(self.base_angle))
            x_diff = y_diff / m_triangle - y_diff / m_self
            self.x = (triangle.get_right_top_vertex()[0]
                      - triangle.get_right_top_vertex()[1] / np.tan(np.deg2rad(self.base_angle))
                      - x_diff)

    def get_right_top_vertex(self):
        return np.array([self.x + self.right_side * np.cos(np.deg2rad(180 - self.base_angle - self.rotation_angle)),
                         self.right_side * np.sin(np.deg2rad(180 - self.base_angle - self.rotation_angle))])

    def get_left_top_vertex(self):
        return np.array([self.x + self.left_side * np.cos(np.deg2rad(180 - self.rotation_angle)),
                         self.left_side * np.sin(np.deg2rad(180 - self.rotation_angle))])

    def get_base_vertex(self):
        return np.array([self.x, 0])

    def get_vertices(self):
        return self.get_base_vertex(), self.get_left_top_vertex(), self.get_right_top_vertex()

    def intersects(self, p1, p2):
        return intersects(np.array([self.x, 0]), self.get_right_top_vertex(), p1, p2)

    def plot(self):
        points = [np.array([self.x, 0]), self.get_right_top_vertex(), self.get_left_top_vertex()]
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        plt.fill(x, y)

    def print(self):
        p1, p2, p3 = self.get_vertices()
        p1, p2, p3 = np.around(p1, 3), np.around(p2, 3), np.around(p3, 3)
        p = 'P({}|{})'.format(p1[0], p1[1])
        q = 'Q({}|{})'.format(p2[0], p2[1])
        r = 'R({}|{})'.format(p3[0], p3[1])
        output = '{:3}:    {:20} {:20} {:20}'.format(self.ID, p, q, r)
        print(output)


# Berechnet den Winkel an b, in grad
def calculate_angle(a, b, c):
    return np.rad2deg(np.arccos(np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))))


# Bestimmt, ob die Geradensegmente p1p2 und q1q2 sich auf eine problematische Weise schneiden
def intersects(p1, p2, q1, q2, recall=True):
    w = np.subtract(p1, q1)
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    v_perp = np.subtract(q2, q1).dot(rotation_matrix)
    numerator = (v_perp * -1).dot(w)
    denominator = v_perp.dot(np.subtract(p2, p1))

    if denominator == 0:
        return False
    else:
        result = numerator / denominator
        if math.isclose(result, 0, abs_tol=1e-6):
            result = 0
        elif math.isclose(result, 1, abs_tol=1e-6):
            result = 1
        if recall:
            cond2 = intersects(q1, q2, p1, p2, recall=False)
            return 0 < result < 1 and cond2
        else:
            return 0 < result < 1


def remove_by_id(triangles, triangle):
    for t in triangles:
        if t.ID == triangle.ID:
            triangles.remove(t)
            break


if __name__ == '__main__':
    main()
