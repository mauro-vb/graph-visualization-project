import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import time
from collections import deque, defaultdict
import math
from networkx.drawing.nx_agraph import graphviz_layout

def angle_compatibility(P, Q):

    # Convert to vectors
    P_vector = np.array(P[1]) - np.array(P[0])
    Q_vector = np.array(Q[1]) - np.array(Q[0])


    dot_product = np.dot(P_vector, Q_vector)

    norm_P = np.linalg.norm(P_vector)

    norm_Q = np.linalg.norm(Q_vector)

    cos_alpha = dot_product / (norm_P * norm_Q)

    Ca = abs(cos_alpha)

    return Ca

def scale_compatibility(P, Q):

    # edge lengths
    length_P = np.linalg.norm(np.array(P[1]) - np.array(P[0]))
    length_Q = np.linalg.norm(np.array(Q[1]) - np.array(Q[0]))

    avg_length = (length_P + length_Q) / 2

#     Cs = 2 / (avg_length*min(length_P, length_Q) + max(length_P, length_Q)/avg_length)
    Cs = 1 / (max(length_P, length_Q)/min(length_P, length_Q))

    return Cs

def distance_compatibility(P, Q):
    P_m = (np.array(P[1]) + np.array(P[0]))/2
    Q_m = (np.array(Q[1]) + np.array(Q[0]))/2

    length_P = np.linalg.norm(np.array(P[1]) - np.array(P[0]))
    length_Q = np.linalg.norm(np.array(Q[1]) - np.array(Q[0]))
    avg_length = (length_P + length_Q) / 2

    Cp = avg_length/(avg_length + np.linalg.norm(P_m - Q_m))
    return Cp

def project_point_onto_line(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unitvec)
    return line_start + projection_length * line_unitvec

def visibility_compatibility(P, Q):
    P_m = (np.array(P[1]) + np.array(P[0]))/2
    Q_m = (np.array(Q[1]) + np.array(Q[0]))/2

    I_0 = project_point_onto_line(np.array(Q[0]), np.array(P[0]), np.array(P[1]))
    I_1 = project_point_onto_line(np.array(Q[1]), np.array(P[0]), np.array(P[1]))
    I_m = (I_0 + I_1) / 2

    length_I = np.linalg.norm(I_0 - I_1)

    vis_P_Q = max(1 - 2 * np.linalg.norm(np.array(P_m) - np.array(I_m)) / length_I, 0)

    # for vis(Q, P)
    J_0 = project_point_onto_line(np.array(P[0]), np.array(Q[0]), np.array(Q[1]))
    J_1 = project_point_onto_line(np.array(P[1]), np.array(Q[0]), np.array(Q[1]))
    J_m = (J_0 + J_1) / 2

    length_J = np.linalg.norm(J_0 - J_1)

    vis_Q_P = max(1 - 2 * np.linalg.norm(np.array(Q_m) - np.array(J_m)) / length_J, 0)

    return min(vis_P_Q, vis_Q_P)

def Ce(P, Q):
    Ce = angle_compatibility(P, Q) * scale_compatibility(P, Q) * distance_compatibility(P, Q) * visibility_compatibility(P, Q)
    return Ce

def subdivide_edge(position, n_points):
    if n_points < 2:
        raise ValueError("Number of points must be at least 2 to subdivide an edge.")

    P0 = position[0]
    P1 = position[1]
    return [tuple(P0 + i/(n_points-1) * (P1 - P0)) for i in range(n_points)]

def edge_bundling(graph, n0, C, I, s, kP):
    B = {}
    for edge_tuple, edge in graph.edges.items():
        if edge.fig_coordinates:
            B[edge_tuple] = subdivide_edge(edge.fig_coordinates, n0)
        else:
            B[edge_tuple] = subdivide_edge((edge.circle1.center,edge.circle2.center), n0)

    c = 0
    max_cycles = C  # Maximum number of cycles to prevent infinite loop

    while c < max_cycles:
        print(f"Cycle {c+1}/{max_cycles}")
        t = 0
        while t < I:
            print(f"  Iteration {t+1}/{I} in cycle {c+1}")
            for edge_tuple, control_points in list(B.items()):
                for i in range(1, len(control_points) - 1):

                    P_i = np.array(control_points[i])
                    P_i_minus_1 = np.array(control_points[i - 1])
                    P_i_plus_1 = np.array(control_points[i + 1])

                    F_spring = kP * (np.linalg.norm(P_i_minus_1 - P_i) + np.linalg.norm(P_i - P_i_plus_1))
                    F_total = F_spring

                    for other_edge, other_control_points in B.items():
                        if other_edge != edge_tuple:
                            for j in range(1, len(other_control_points) - 1):
                                Q_j = np.array(other_control_points[j])
                                compatibility_score = Ce((P_i_minus_1, P_i_plus_1), (other_control_points[j - 1], other_control_points[j + 1]))
                                distance = np.linalg.norm(P_i - Q_j) * 10

                                if distance > 0.001 and compatibility_score > 0.55:
                                    F_electrostatic = compatibility_score / distance
                                    F_total += F_electrostatic * (Q_j - P_i)

                    B[edge_tuple][i] = tuple(np.array(control_points[i]) + s * F_total)

            t += 1
            if t >= I:
                c += 1

                # Increase the number of control points for next iteration by subdividing current control points
                for edge_tuple in list(B.keys()):
                    new_control_points = []
                    for i in range(len(B[edge_tuple]) - 1):
                        mid_point = tuple((np.array(B[edge_tuple][i]) + np.array(B[edge_tuple][i + 1])) / 2)
                        new_control_points.extend([B[edge_tuple][i], mid_point])
                    new_control_points.append(B[edge_tuple][-1])
                    B[edge_tuple] = new_control_points

                n0 = len(B[edge_tuple])  # Update n0 to reflect new number of control points
                s /= 2  # Halving the step size for more refined control point adjustments
                I  = int(I * (2/3))
                print(f"  Updated control points, reduced iteration number to {I} and halved step size to {s}.")


        if c >= max_cycles:
            print("Reached maximum cycle limit.")
            break

    print("Edge bundling completed.")
    return B
