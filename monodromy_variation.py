import subprocess
import time
import os
import tkinter as tk
from tkinter import StringVar, Entry, Button, Canvas, messagebox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from io import BytesIO
from PIL import Image, ImageTk
from math import gcd
from fractions import Fraction

# =============================================================================
# Global Constants for Singular Commands
# =============================================================================

start_ring_commands = [
    'LIB "all.lib";',
    'ring r = 0, (x, y), ds;',
]

# =============================================================================
# Helper Functions for 2x2 Matrix Operations and Arithmetic
# (These replace SageMath’s Matrix functionality.)
# =============================================================================

def matrix2x2(a, b, c, d):
    """
    Create a 2x2 matrix represented as a list of lists with Fraction entries.
    """
    return [[Fraction(a), Fraction(b)], [Fraction(c), Fraction(d)]]

def mat_mult(A, B):
    """
    Multiply two 2x2 matrices A and B.
    """
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ]

def rational_part(i):
    """
    Return the fractional part of a Fraction.
    """
    return i - i.numerator // i.denominator

def frac(a, b):
    """
    Create a Fraction from a and b. If b is already a Fraction, perform division.
    """
    if isinstance(b, Fraction):
        return Fraction(a) / b
    else:
        return Fraction(int(a), int(b))

# =============================================================================
# Singular Interface Functions
# (These functions start Singular, send commands, and capture output.)
# =============================================================================

def start_singular():
    """
    Start a Singular process and initialize it.
    """
    init = ['Singular', '-q']
    s = subprocess.Popen(
        init,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    # Write a test print command to ensure Singular is responsive.
    s.stdin.write("print('GO');\n")
    os.read(s.stdout.fileno(), 8192).decode()
    return s

def singular_talk(msg, s):
    """
    Send a command to Singular and return its output.
    """
    s.stdin.write(msg + 'print("");\n')
    time.sleep(0.1)
    response = os.read(s.stdout.fileno(), 8192).decode()
    return response

# =============================================================================
# Utility Functions for Matrix Conversion
# =============================================================================

def is_relevant_row(row):
    """
    Check whether the row string starts with a number.
    Used to filter out non-data lines from Singular output.
    """
    before_comma = row.split(",")[0]
    try:
        int(before_comma)
        return True
    except ValueError:
        return False

def string_to_numpy_matrix(matrix_string):
    """
    Convert a string representation of a matrix (from Singular) to a NumPy array.
    """
    _rows = matrix_string.strip().split('\n')
    rows = [row for row in _rows if is_relevant_row(row)]
    # If only one row is detected, split by commas accordingly.
    if len(rows) == 1:
        rows = rows[0].split(',')
        matrix = [list(filter(None, map(str.strip, row.split(',')))) for row in rows]
    else:
        matrix = [list(filter(None, map(str.strip, row.split(',')))) for row in rows]
    np_matrix = np.array(matrix, dtype=int)
    return np_matrix

# =============================================================================
# Graph Drawing Functions
# =============================================================================

def draw_resolution_graph(G, H, mult=False, order=False, euler=False, canonical=False):
    """
    Draw the resolution graph G and its subgraph H (arrow nodes) using matplotlib.
    Optional parameters control whether to draw multiplicity, order, Euler number,
    or canonical discrepancy labels.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G)
    if mult:
        pos_mults = {node: (coords[0], coords[1] + 0.095) for node, coords in pos.items()}
        labels = nx.get_node_attributes(G, 'multiplicity')
        custom_node_mults = {node: f"$m_{{{node}}}$: {attr}" for node, attr in labels.items()}
        nx.draw_networkx_labels(G, pos_mults, labels=custom_node_mults, ax=ax)
    if order:
        pos_ords = {node: (coords[0], coords[1] + 0.095) for node, coords in pos.items()}
        labels = nx.get_node_attributes(G, 'order')
        custom_node_ords = {node: f"$o_{{{node}}}$: {attr}" for node, attr in labels.items()}
        nx.draw_networkx_labels(G, pos_ords, labels=custom_node_ords, ax=ax)
    if euler:
        pos_eul = {node: (coords[0], coords[1] - 0.095) for node, coords in pos.items()}
        node_euls = nx.get_node_attributes(G, 'euler_number')
        custom_node_euls = {node: f"$e_{{{node}}}$: {attr}" for node, attr in node_euls.items()}
        nx.draw_networkx_labels(G, pos_eul, labels=custom_node_euls, ax=ax)
    if canonical:
        pos_can = {node: (coords[0] + 0.1, coords[1]) for node, coords in pos.items()}
        labels = nx.get_node_attributes(G, 'discrepancies')
        custom_node_can = {node: f"$d_{{{node}}}$: {attr}" for node, attr in labels.items()}
        nx.draw_networkx_labels(G, pos_can, labels=custom_node_can, ax=ax)
    # Draw main graph G.
    nx.draw(G, pos, edge_color='red', width=1, linewidths=2,
            with_labels=True, node_size=250, node_color='skyblue', node_shape='o', ax=ax)
    # Draw subgraph H with different styling.
    nx.draw(H, pos, edge_color='red', width=1, linewidths=2,
            with_labels=True, node_size=350, node_shape='o',
            node_color="none", edgecolors="black", ax=ax)
    ax.axis('off')
    return fig

def plot_graph():
    """
    Extracts a polynomial from a global tkinter variable, runs the resolution
    computation, draws the graph, and displays it on a tkinter canvas.
    """
    polynomial = poly_var.get()
    if not polynomial:
        messagebox.showerror("Error", "Please enter a valid polynomial!")
        return
    G_local, H_sub, Up_sub, multiplicity_matrix = run_polynomial(polynomial)
    fig = draw_resolution_graph(G_local, H_sub)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    canvas.img = img_tk  # Keep a reference to avoid garbage collection.
    canvas.create_image(300, 300, anchor="center", image=img_tk)
    buf.close()

def place_holder():
    """
    Placeholder for additional tkinter callback functions.
    """
    pass

# =============================================================================
# Graph Utility Functions for Resolution Data
# =============================================================================

def discrepancies(incidence_matrix, res_graph, arrows):
    """
    Compute the discrepancy vector by constructing and solving a linear system.
    """
    H_sub = arrows
    G_local = res_graph
    num_nodes, _ = incidence_matrix.shape
    disc_matrix = []
    indep_vector = []
    for i in range(num_nodes):
        new_vector = [0] * num_nodes
        neighbors = [j for j in G_local.neighbors(i) if j not in H_sub.nodes()]
        k = len(neighbors)
        if i in H_sub.nodes():
            indep_vector.append(1)
            new_vector[i] = 1
        else:
            indep_vector.append(2 - k)
            new_vector[i] = -incidence_matrix[i][i]
            for j in G_local.neighbors(i):
                if j not in H_sub.nodes():
                    new_vector[j] = -1
                else:
                    new_vector[j] = 0
        disc_matrix.append(new_vector)
    matrix = np.array(disc_matrix)
    ind = np.transpose(np.array(indep_vector))
    solutions = (np.rint(np.matmul(np.linalg.inv(matrix), ind))).astype(int)
    return solutions

def define_weighted_dual_graph(incidence_matrix, total_mult, inv_incidence_matrix):
    """
    Define the weighted dual graph using the incidence matrix and multiplicities.
    This function also computes various invariants such as Euler numbers,
    discrepancies, and maximum cycles.
    """
    num_nodes, _ = incidence_matrix.shape
    G_local = nx.Graph()
    # Add edges based on the incidence matrix.
    for i in range(num_nodes):
        edges = [j for j in range(num_nodes) if incidence_matrix[j][i] == 1 and j != i]
        for j in edges:
            G_local.add_edge(i, j)
    # Nodes with self-intersection -1 are taken as arrow nodes.
    arrows = [i for i in range(num_nodes) if incidence_matrix[i][i] == -1]
    H_sub = G_local.subgraph(arrows)
    Up_vertices = []
    for a in arrows:
        Up_vertices += [j for j in nx.shortest_path(G_local, a, 0)]
    Up_vertices = list(set(Up_vertices))
    Up_sub = G_local.subgraph(Up_vertices)
    # Adjust Euler numbers and total multiplicities.
    for i in H_sub.nodes:
        incidence_matrix[i][i] = 0
        total_mult[i] = 1
        for j in G_local.neighbors(i):
            incidence_matrix[j][j] += 1
    disc = discrepancies(incidence_matrix, G_local, H_sub)
    for i in range(num_nodes):
        G_local.nodes[i]['multiplicity'] = int(total_mult[i])
        G_local.nodes[i]['euler_number'] = incidence_matrix[i][i]
        G_local.nodes[i]['discrepancies'] = disc[i]
        G_local.nodes[i]['c0'] = int(-inv_incidence_matrix[0][i])
        G_local.nodes[i]['alpha'] = 1
        G_local.nodes[i]['beta'] = 1
        G_local.nodes[i]['c1'] = G_local.nodes[i]['discrepancies'] - G_local.nodes[i]['c0']
        G_local.nodes[i]['varpi'] = 1
    for i in H_sub.nodes:
        G_local.nodes[i]['c0'] = 0
        G_local.nodes[i]['c1'] = 1
    for e in G_local.edges:
        G_local.edges[e]['nik'] = int(abs(G_local.nodes[e[0]]['multiplicity'] *
                                           G_local.nodes[e[1]]['c0'] -
                                           G_local.nodes[e[1]]['multiplicity'] *
                                           G_local.nodes[e[0]]['c0']))
    # Adjust invariants for leaf nodes.
    v_leaves = [l for l, degree in G_local.degree() if degree <= 1 and l not in list(H_sub.nodes()) and l != 0]
    for l in v_leaves:
        path = nx.shortest_path(G_local, l, 0)
        i = 0
        while G_local.degree[path[i]] < 3:
            i += 1
        G_local.nodes[path[i]]['alpha'] = G_local.nodes[path[i]]['multiplicity'] // G_local.nodes[l]['multiplicity']
        G_local.nodes[path[i]]['beta'] = G_local.nodes[path[i - 1]]['multiplicity'] // G_local.nodes[l]['multiplicity']
    return G_local, H_sub, Up_sub

# =============================================================================
# Graph Invariants and Basic Graph Functions
# =============================================================================

def screw_number(G, p, q):
    """
    Calculate the screw number between two nodes p and q in G.
    This is a weighted sum along the unique shortest path.
    """
    path = list(nx.shortest_path(G, p, q))
    screw_coefficient = Fraction(0, 1)
    d = int(gcd(G.nodes[path[0]]['multiplicity'], G.nodes[path[1]]['multiplicity']))
    for i, node in enumerate(path[:-1]):
        screw_coefficient += Fraction(1, (G.nodes[node]['multiplicity'] *
                                            G.nodes[path[i + 1]]['multiplicity']))
    return Fraction(d**2, 1) * screw_coefficient

# =============================================================================
# Resolution Graph Functions (Splicing and Ordering)
# =============================================================================

def splice_graph(G, H):
    """
    Create a splice graph (EN) from the resolution graph G.
    Only nodes of degree not equal to 2 (and the special node 0) are kept.
    Also computes Seifert invariants for nodes.
    """
    EN = nx.Graph()
    EN.add_node(0)
    for i in G.nodes():
        if G.degree[i] != 2:
            EN.add_node(i)
            EN.nodes[i]['multiplicity'] = G.nodes[i]['multiplicity']
    for i in list(EN.nodes()):
        for j in [j for j in list(EN.nodes()) if j > i]:
            path = nx.shortest_path(G, i, j)
            d = True
            for k in range(1, len(path) - 1):
                if G.degree[path[k]] != 2 or path[k] == 0:
                    d = False
            if d:
                EN.add_edge(i, j)
                EN.edges[i, j]['screw_number'] = screw_number(G, i, j)
    for i in EN.nodes():
        EN.nodes[i]['seifert_alpha'] = Fraction(1, 1)
        EN.nodes[i]['seifert_beta'] = Fraction(0, 1)
    for i in [i for i in EN.nodes() if EN.degree[i] == 1 and i not in H.nodes() and i != 0]:
        n = list(EN.neighbors(i))[0]
        path = nx.shortest_path(G, i, n)
        k = len(path)
        EN.nodes[i]['seifert_alpha'] = Fraction(G.nodes[n]['multiplicity'], G.nodes[i]['multiplicity'])
        EN.nodes[n]['seifert_alpha'] = Fraction(G.nodes[n]['multiplicity'], G.nodes[i]['multiplicity'])
        EN.nodes[i]['seifert_beta'] = Fraction(G.nodes[path[k-2]]['multiplicity'], G.nodes[i]['multiplicity'])
        EN.nodes[n]['seifert_beta'] = Fraction(G.nodes[path[k-2]]['multiplicity'], G.nodes[i]['multiplicity'])
    return EN

def leaf(G, H, i):
    """
    Return the unique neighbor of node i (if any) that is a leaf (degree 1),
    not in the subgraph H and not the special node 0.
    """
    l = [j for j in G.neighbors(i) if (j not in H.nodes()) and (G.degree(j) == 1) and (j != 0)]
    if l:
        return l[0]
    else:
        return -1

def ordering(G, H, max_value, i):
    """
    Recursively assign an 'order' to the nodes of the graph G.
    This function updates the node attribute 'order' and returns the next available value.
    """
    G.nodes[i]['order'] = max_value
    max_value += 1
    if leaf(G, H, i) != -1:
        G.nodes[leaf(G, H, i)]['order'] = max_value
        max_value += 1
    L = list(G.neighbors(i))
    L.sort()
    for j in L:
        if j != predecessor(G, i) and j != leaf(G, H, i):
            max_value = ordering(G, H, max_value, j)
    return max_value

def predecessor(G, i):
    """
    Return the predecessor of node i in the unique shortest path from i to 0.
    """
    p = list(nx.shortest_path(G, i, 0))
    if len(p) > 1:
        return p[1]
    else:
        return -1

def greater_neighbors_G_sorted(G, Up, i):
    greater_neigh = [j for j in G.neighbors(i) if j != predecessor(G, i) and j in Up.nodes]
    greater_neigh.sort()
    return greater_neigh

def greater_neighbors_EN_sorted(EN, G, Up, H, i):
    greater_neigh = greater_neighbors_G_sorted(G, Up, i)
    greater_neigh_EN_sorted = []
    for k in range(len(greater_neigh)):
        greater_neigh_EN_sorted.append(further_neighbor(EN,G,H,i,greater_neigh[k]))
    return greater_neigh_EN_sorted
# =============================================================================
# Matrix and Coordinate Functions for Transition Maps
# =============================================================================

def path_matrix(G, i, j):
    """
    Compute a 2x2 matrix associated with the path from node i to node j.
    This is built as a product of elementary matrices along the path.
    """
    r = matrix2x2(0, 1, 1, 0)
    path = nx.shortest_path(G, i, j)
    for idx in range(1, len(path) - 1):
        b = -G.nodes[path[idx]]['euler_number']
        r = mat_mult(matrix2x2(1, b, 0, -1), r)
        r = mat_mult(matrix2x2(0, 1, 1, 0), r)
    return r

# =============================================================================
# Functions for Order and Neighbor Relations
# =============================================================================

def Sik(G, H, i, k):
    """
    Return the list of neighbors of node i whose 'order' is greater than that of node k.
    """
    return [l for l in G.neighbors(i) if G.nodes[l]['order'] > G.nodes[k]['order']]

def Mik(G, H, i, k):
    """
    Compute a sum over multiplicities along a path from k to 0.
    """
    if i not in G.neighbors(k) or G.nodes[i]['order'] > G.nodes[k]['order']:
        print('bad shit')
    path_to_zero = list(nx.shortest_path(G, k, 0))
    return sum([G.nodes[c]['multiplicity'] for r in range(len(path_to_zero) - 1)
                for c in Sik(G, H, path_to_zero[r + 1], path_to_zero[r])])

def sign_aik(G, H, i, k):
    """
    Compute the sign associated with nodes i and k based on Mik (that is, the
    sign of the coefficient a_ik).
    """
    return (-1) ** Mik(G, H, i, k)

def nik(G, H, i, k):
    """
    Compute an invariant (called 'n_ik') based on multiplicities and c0.
    """
    if i == -1 or k == -1:
        #return sum([nik(G, H, 0, l) for l in G.neighbors(0)])
        return G.nodes[0]['multiplicity']
    return abs(G.nodes[i]['multiplicity'] * G.nodes[k]['c0']
               - G.nodes[k]['multiplicity'] * G.nodes[i]['c0'])

# =============================================================================
# Functions for Building Blocks and Matrix Entry Indices
# =============================================================================

def size_block(EN, G, Up, H, i):
    """
    Determine the size of the block corresponding to node i.
    """
    greater_neigh = greater_neighbors_EN_sorted(EN,G,Up,H,i)
    return G.nodes[i]['multiplicity'] * (len(greater_neigh) - 1) + int(
        G.nodes[i]['multiplicity'] * (1 - Fraction(1, EN.nodes[i]['seifert_alpha'])))

def size_monodromy(EN, G, Up, H):
    """
    Determine the size of the monodromy matrix (which coincides with the size of the variation matrix).
    """
    return sum([size_block(EN,G, Up,H,i) for i in EN.nodes() if EN.degree(i)!=1])

def entry_mat(EN, G, Up, H, i, a, b):
    """
    Calculate the entry index in the block matrix for node i given parameters a and b.
    """
    prev_length = sum(size_block(EN, G, Up, H, j) for j in EN.nodes()
                      if G.nodes[i]['order'] > G.nodes[j]['order'] and EN.degree(j) != 1)
    m_i = G.nodes[i]['multiplicity']
    if a == 0:
        return prev_length + b - 1
    return prev_length + int(m_i * (1 - Fraction(1, EN.nodes[i]['seifert_alpha']))) + (a - 1) * m_i + b - 1

def ordered_nodes(EN,G,H):
    real_nodes = [k for k in EN.nodes() if EN.degree(k) != 1 or k ==0]
    def a(i):
        return G.nodes[i]['order']
    real_nodes.sort(key = a)
    return real_nodes

def entry_mat_inv(EN, G,Up, H, p):
    """
    Returns the entries in the i,a,b
    """
    ord_nodes = ordered_nodes(EN,G,H)
    i = 0
    while (entry_mat(EN, G, Up, H, ord_nodes[i], 0, 1) + size_block(EN,G, Up,H,ord_nodes[i])) <= p:
        i+=1
    a = 0
    while entry_mat(EN, G, Up, H, ord_nodes[i], a+1, 1) <= p:
        a+=1
    b=1
    while entry_mat(EN, G, Up, H, ord_nodes[i], a, b) < p:
        b+=1
    return ord_nodes[i], a, b
# =============================================================================
# Delta Functions: Coordinate Changes Between Charts
# =============================================================================

def Delta_hat(G, H, i, k, beta_ik, sign=1):
    """
    Compute the modified beta coordinate (Delta_hat) for a transition.
    """
    beta_ik = rational_part(beta_ik)
    j = predecessor(G, i)
    if Fraction(0, 1) < beta_ik < Fraction(1, 2):
        beta_i = beta_ik * frac(nik(G, H, i, k), nik(G, H, i, j)) + \
                 sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)])
    elif Fraction(1, 2) < beta_ik < Fraction(1, 1):
        beta_i = 1 - ((1 - beta_ik) * frac(nik(G, H, i, k), nik(G, H, i, j)) +
                      sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)]))
    elif beta_ik == Fraction(1, 2) and sign == 1:
        beta_i = beta_ik * frac(nik(G, H, i, k), nik(G, H, i, j)) + \
                 sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)])
    elif beta_ik == Fraction(1, 2) and sign == -1:
        beta_i = 1 - ((1 - beta_ik) * frac(nik(G, H, i, k), nik(G, H, i, j)) +
                      sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)]))
    elif beta_ik == Fraction(0, 1) and sign == 1:
        beta_i = beta_ik * frac(nik(G, H, i, k), nik(G, H, i, j)) + \
                 sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)])
    elif beta_ik == Fraction(0, 1) and sign == -1:
        beta_i = 1 - (beta_ik * frac(nik(G, H, i, k), nik(G, H, i, j)) +
                      sum([frac(nik(G, H, i, l), 2 * nik(G, H, i, j)) for l in Sik(G, H, i, k)]))
    tbeta_i = - beta_i
    return rational_part(tbeta_i)

def Delta_hat_inv(G, Up, H, i, tbeta_i):
    """
    Inverse of Delta_hat: recovers beta_ik from the modified beta coordinate. At the moment it works for generic T.
    """
    greater_neigh = greater_neighbors_G_sorted(G, Up, i)
    beta_i = rational_part(-tbeta_i)
    lengths = [Fraction(0, 1)]
    pred = predecessor(G, i)

    for l in range(len(greater_neigh)):
        lengths.append(frac(nik(G, H, i, greater_neigh[l]), (2 * nik(G, H, i, pred))))
    coordinates = [Fraction(0, 1)]
    for l in range(len(greater_neigh)):
        coordinates.append(coordinates[-1] + lengths[-1 - l])
    if 0 <= beta_i < Fraction(1, 2):
        t = max([l for l in range(len(coordinates)) if beta_i >= coordinates[l]])
        target = greater_neigh[-1 - t]
        beta_ik = (beta_i - coordinates[t]) / (2 * (coordinates[t + 1] - coordinates[t]))
    elif Fraction(1, 2) < beta_i < 1:
        t = max([l for l in range(len(coordinates)) if 1 - beta_i >= coordinates[l]])
        target = greater_neigh[-1 - t]
        beta_ik = 1 - (1 - beta_i - coordinates[t]) / (2 * (coordinates[t + 1] - coordinates[t]))
    else:
        target = greater_neigh[0] if greater_neigh else -1
        beta_ik = Fraction(0, 1)
    return target, rational_part(beta_ik)

def Delta(G, H, i, k, alpha_i, beta_ik, sign=1):
    """
    Compute the transition (Delta) mapping for the coordinates at a node.
    Returns a pair [new_alpha, new_beta].
    """
    alpha_i = rational_part(alpha_i)
    beta_ik = rational_part(beta_ik)
    m_i = G.nodes[i]['multiplicity']
    m_k = G.nodes[k]['multiplicity']
    r = [Fraction(0, 1), Fraction(0, 1)]
    j = predecessor(G, i)
    m_j = 0 if j == -1 else G.nodes[j]['multiplicity']
    sum_sign = 0
    beta_sum_sign = 1
    beta_coeff = beta_ik
    if beta_ik > 0 and beta_ik < Fraction(1,2):
        sum_sign = 1
    elif beta_ik > Fraction(1,2) and beta_ik < Fraction(1,1):
        sum_sign = -1
        beta_coeff = -(1 - beta_ik)
    elif beta_ik == 0:
        sum_sign = sign
    elif beta_ik == Fraction(1,2):
        sum_sign = sign
        if sign == -1:
            beta_coeff = -(1 - beta_ik)

    b_i = G.nodes[i]['euler_number']
    r[0] = alpha_i + sum_sign*sum([frac(G.nodes[l]['multiplicity'], 2 * m_i)
                          + (frac(m_j,m_i) - b_i) * frac(nik(G, H, i, l), 2 * nik(G, H, i, j))
                          for l in Sik(G, H, i, k)]) \
           + 2 * beta_coeff * (frac(m_k, 2 * m_i)
                            + (frac(m_j,m_i)  - b_i) *frac(nik(G, H, i, k), 2 * nik(G, H, i, j)))
    r[1] = Delta_hat(G, H, i, k, beta_ik, sign)
    r[0] = rational_part(r[0] - b_i * r[1])
    r[1] = rational_part(r[1])
    return r

def Delta_inv(G, Up, H, i, talpha_i, tbeta_i, sign=1):
    """
    Inverse of Delta: given modified coordinates, recover the original ones.
    Returns the neighbor k and the coordinates (alpha, beta).
    """
    j = predecessor(G, i)
    k, r_beta_ik = Delta_hat_inv(G, Up, H, i, tbeta_i)
    m_i = G.nodes[i]['multiplicity']
    m_k = G.nodes[k]['multiplicity']
    b_i = G.nodes[i]['euler_number']
    alpha_i = rational_part(talpha_i + b_i * tbeta_i)
    sum_sign = 0
    beta_sum_sign = 1
    beta_coeff = r_beta_ik
    if r_beta_ik > 0 and r_beta_ik < Fraction(1,2):
        sum_sign = 1
    elif r_beta_ik > Fraction(1,2) and r_beta_ik < Fraction(1,1):
        sum_sign = -1
        beta_coeff = - (1 - r_beta_ik)
    elif r_beta_ik == 0:
        sum_sign = sign
    elif r_beta_ik == Fraction(1,2):
        sum_sign = sign
        if sign == -1:
            beta_coeff = -(1 - r_beta_ik)
    m_j = 0 if j == -1 else G.nodes[j]['multiplicity']
    r_alpha_i = alpha_i - sum_sign*sum([frac(G.nodes[l]['multiplicity'], 2 * m_i)
                        + (frac(m_j,m_i)- b_i) * frac(nik(G, H, i, l), 2 * nik(G, H, i, j))
                                for l in Sik(G, H, i, k)])  \
                           - 2 * beta_coeff * (frac(m_k, 2 * m_i) + (frac(m_j,m_i) - b_i) *
                                            frac(nik(G, H, i, k), 2 * nik(G, H, i, j)))
    return k, rational_part(r_alpha_i), rational_part(r_beta_ik)

# =============================================================================
# Functions for Monodromy Blocks and Neighbor Ordering
# =============================================================================

def monodromy_block_Bia(G, EN, H, i, a):
    """
    Build a block (matrix) associated with node i used in the monodromy matrix.
    Two cases are considered depending on the parameter a.
    """
    m = G.nodes[i]['multiplicity']
    if a == 0:
        a_i = int(EN.nodes[i]['seifert_alpha'])
        block_size = int(m - m / a_i)
        block = np.zeros((block_size, block_size), dtype=int)
        for j in range(block_size - 1):
            block[j + 1, j] = 1
        for j in range(a_i - 1):
            #block[j * (m // a_i), block_size - 1] = -1
            block[0,block_size - j * (m // a_i) - 1 ] = -1
    else:
        block_size = m
        block = np.zeros((block_size, block_size), dtype=int)
        for j in range(block_size - 1):
            block[j + 1, j] = 1
        block[0, -1] = 1
    return block

def variation_block_Bia(G, EN, H, i, a):
    """
    Build a block (matrix) associated with node i used in the variation matrix.
    Two cases are considered depending on the parameter a. HIGHLY NON FINISHED
    """
    m = G.nodes[i]['multiplicity']
    if a == 0:
        a_i = int(EN.nodes[i]['seifert_alpha'])
        block_size = int(m - m / a_i)
        block = np.zeros((block_size, block_size), dtype=int)
        for j in range(block_size - 1):
            block[j + 1, j] = 1
        for j in range(a_i - 1):
            block[j * (m // a_i), block_size - 1] = -1
    else:
        block_size = m
        block = np.zeros((block_size, block_size), dtype=int)
        for j in range(block_size - 1):
            block[j + 1, j] = 1
        block[0, -1] = 1
    return block

def ath_neighbor(G, Up, H, i, a):
    """
    Return the ath neighbor of node i (ordered by increasing index) that is
    neither the leaf nor the predecessor.
    """
    greater_neigh = greater_neighbors_G_sorted(G, Up, i)
    return greater_neigh[a - 1]

def ath_neighbor_inv(G, Up, H, i, k):
    """
    Given a neighbor k of i, return its position (index) among the ordered list
    of valid neighbors.
    """
    greater_neigh = greater_neighbors_G_sorted(G, Up, i)
    if k in greater_neigh:
        return greater_neigh.index(k) + 1
    else:
        return -1

def further_neighbor(EN,G,H, i, k):
    """
    k is a neighbor of i
    """
    r = k
    s = i
    while r not in EN.nodes and r not in H.nodes:
        new = [a for a in G.neighbors(r) if a != s]
        s = r
        r = new[0]
    return r

def ath_neighbor_EN(EN,G,Up,H,i,a):
    local_neigh = ath_neighbor(G,Up,H,i,a)
    return further_neighbor(EN,G,H,i,local_neigh)

def ath_neighbor_inv_EN(EN,G,Up,H,i,k):
    """
    i and k are two nodes connected by a bamboo
    """
    spath = nx.shortest_path(G, i, k)
    return ath_neighbor_inv(G, Up, H, i, spath[1])
# =============================================================================
# Starting Alpha Functions for Coordinate Initialization
# =============================================================================

def starting_alpha_petri_stable(G, Up, H, i, b, T):
    """
    Compute the starting alpha coordinate for the 'petri' case. The output is in
    alpha_i coordinate.
    """
    m_i = G.nodes[i]['multiplicity']
    not_invariant = [k for k in G.neighbors(i) if k != predecessor(G, i) and k not in Up.nodes]
    k = not_invariant[0]
    m_k = G.nodes[k]['multiplicity']
    alpha = rational_part(T - Fraction((Mik(G, H, i, k) % 2), 2) - Fraction(m_k, 2))
    alpha = alpha / m_i
    return rational_part(alpha + Fraction(b - 1, m_i))

def starting_alpha(G, Up, H, i, a, b, T):
    """
    Compute the starting alpha coordinate for the standard case.
    """
    ath = ath_neighbor(G, Up, H, i, a)
    m_i = G.nodes[i]['multiplicity']
    alpha = rational_part(T  + Fraction((Mik(G, H, i, ath) % 2), 2))
    alpha = rational_part(alpha) / m_i
    return rational_part(alpha + Fraction(b - 1, m_i))

def arg_f(G, Up, H, i, talpha, tbeta):
    m_i = G.nodes[i]['multiplicity']
    j = predecessor(G, i)
    m_j = 0 if j == -1 else G.nodes[j]['multiplicity']
    return rational_part(m_i*talpha + m_j*tbeta + frac(Mik(G, H, j, i), 2))

# =============================================================================
# Functions for Changing Coordinates Between Charts
# =============================================================================

def change_of_coords(G, H, i, k, alpha_i, beta_i):
    """
    Change coordinates from node i to k using the path matrix.
    """
    PM = path_matrix(G, i, k)
    py_PM = [[int(PM[0][0]), int(PM[0][1])],
             [int(PM[1][0]), int(PM[1][1])]]
    vect = [py_PM[0][0] * alpha_i + py_PM[0][1] * beta_i,
            py_PM[1][0] * alpha_i + py_PM[1][1] * beta_i]
    return [rational_part(vect[0]), rational_part(vect[1])]

def alpha_beta_coords_S(EN, G, Up, H, i, a, b, T):
    """
    Compute the stable manifold coordinates (alpha and beta) for a node in EN.
    """
    alpha_beta = [[], []]
    geodesic_to_zero = nx.shortest_path(EN, i, 0)
    if a == 0:
        not_invariant = [j for j in G.neighbors(i) if j != predecessor(G, i) and j not in Up.nodes]
        k = not_invariant[0]
        m_i = G.nodes[i]['multiplicity']
        m_k = G.nodes[k]['multiplicity']
        m_ik = gcd(m_i, m_k)
        s_alpha = rational_part(starting_alpha_petri_stable(G, Up, H, i, b, T) + frac(G.nodes[i]['euler_number'], 2))
        alpha_beta[0].append([s_alpha, Fraction(1, 2)])
        b_p = ((b - 1) % m_ik) + 1 + m_i - m_ik
        s_alpha_p = rational_part(starting_alpha_petri_stable(G, Up, H, i, b_p, T) + frac(G.nodes[i]['euler_number'], 2))
        alpha_beta[1].append([s_alpha_p, Fraction(1, 2)])
    elif a != 0:
        ath = ath_neighbor(G,Up, H, i, a)
        s_alpha = starting_alpha(G,Up, H, i, a, b, T)
        alpha_beta[0].append(Delta(G, H, i, ath, s_alpha, Fraction(0, 1), -1))
        alpha_beta[1].append(Delta(G, H, i, ath, s_alpha, Fraction(0, 1), 1))
    for j in range(len(geodesic_to_zero) - 1):
        vect_neg = change_of_coords(G, H, geodesic_to_zero[j], geodesic_to_zero[j + 1],
                                     alpha_beta[0][j][0], alpha_beta[0][j][1])
        vect_pos = change_of_coords(G, H, geodesic_to_zero[j], geodesic_to_zero[j + 1],
                                     alpha_beta[1][j][0], alpha_beta[1][j][1])
        aux_vertex = nx.shortest_path(G, geodesic_to_zero[j + 1], geodesic_to_zero[j])[1]
        alpha_beta[0].append(Delta(G, H, geodesic_to_zero[j + 1], aux_vertex, vect_neg[0], vect_neg[1]))
        alpha_beta[1].append(Delta(G, H, geodesic_to_zero[j + 1], aux_vertex, vect_pos[0], vect_pos[1]))
    return alpha_beta

def alpha_beta_coords_U(EN, G, Up, H, j, c, d, T):
    """
    Compute the unstable manifold coordinates (alpha and beta) for a node.
    """
    alpha_beta = [[], []]
    greater_neighbors_EN = greater_neighbors_EN_sorted(EN,G,Up,H,j)
    if c == 0:
        not_invariant = [k for k in G.neighbors(j) if k != predecessor(G, j) and k not in Up.nodes]
        k = not_invariant[0]
        ell = ath_neighbor_EN(EN,G,Up,H,j,1)
        m_j = G.nodes[j]['multiplicity']
        seifert_alpha = EN.nodes[j]['seifert_alpha']
        m_k = G.nodes[k]['multiplicity']
        m_jk = gcd(m_j, m_k)
        m_j_p = frac(m_j, m_jk)
        m_k_p = frac(m_k, m_jk)
        s_alpha = starting_alpha_petri_stable(G, Up, H, j, d, T)
        alpha_beta_pos = change_of_coords(G, H, j, ell, s_alpha - m_k_p / (2 * seifert_alpha),
                                           frac(1,2))
        alpha_beta_neg = change_of_coords(G, H, j, ell, s_alpha + m_k_p / (2 * seifert_alpha),
                                           frac(1,2))

        alpha_beta[0].append([ell, alpha_beta_neg])
        alpha_beta[1].append([ell, alpha_beta_pos])
    elif c != 0:
        ath_neighbor_pos = greater_neighbors_EN[c]
        ath_neighbor_neg = greater_neighbors_EN[c - 1]
        s_alpha = starting_alpha(G, Up, H, j, c, d, T)
        alpha_beta_pos = change_of_coords(G, H, j, ath_neighbor_pos, s_alpha, Fraction(1, 2))
        alpha_beta_neg = change_of_coords(G, H, j, ath_neighbor_neg, s_alpha, Fraction(0, 1))
        alpha_beta[0].append([ath_neighbor_neg, alpha_beta_neg])
        alpha_beta[1].append([ath_neighbor_pos, alpha_beta_pos])
    while alpha_beta[0][-1][0] not in H.nodes():
        l = alpha_beta[0][-1][0]
        k, alpha_j, beta_jk = Delta_inv(G, Up, H, l, alpha_beta[0][-1][1][0], alpha_beta[0][-1][1][1])
        n_k = further_neighbor(EN,G,H,l,k)
        alpha_beta[0].append([n_k, change_of_coords(G, H, l, n_k, alpha_j, beta_jk)])
    while alpha_beta[1][-1][0] not in H.nodes():
        l = alpha_beta[1][-1][0]
        k, alpha_j, beta_jk = Delta_inv(G,Up, H, l, alpha_beta[1][-1][1][0], alpha_beta[1][-1][1][1])
        n_k = further_neighbor(EN,G,H,l,k)
        alpha_beta[1].append([n_k, change_of_coords(G, H, l, n_k, alpha_j, beta_jk)])
    return alpha_beta

def coordinate_intersection(EN, G, H, i, a_s, b_s, a_u, b_u, ded = False, s = 0):
    """
    Compute the intersection number of stable and unstable manifolds at node i.
    """
    m_i = G.nodes[i]['multiplicity']
    a_g, b_g = a_s + Fraction(1, m_i), b_s
    a_v = a_g - a_u
    b_v = b_g - b_u
    j_G = predecessor(G, i)
    m_j_G = G.nodes[j_G]['multiplicity']
    g_val = int(gcd(m_i, m_j_G))
    if rational_part(frac((m_i * a_v + m_j_G * b_v), g_val)) != 0:
        return 0
    else:
        j_EN = predecessor(EN, i)
        sn = screw_number(G, i, j_EN)
        m_i_p = int(Fraction(m_i, g_val))
        m_j_G_p = int(Fraction(m_j_G, g_val))
        c = pow(-m_j_G_p, -1, m_i_p)
        d_val = frac((1 + c * m_j_G_p), m_i_p)
        if rational_part(-(c * a_v + d_val * b_v)) < sn / g_val and 0 < rational_part(-(c * a_v + d_val * b_v)):
            return 1
        elif rational_part(-(c * a_v + d_val * b_v)) == sn / g_val and ded == True:
            return s
        elif  0 == rational_part(-(c * a_v + d_val * b_v)) and ded == True:
            return 1 - s
        else:
            return 0

def intersection_stable_unstable(EN, G, Up, H, i, a, b, j, c, d, T):
    """
    Compute the algebraic intersection number between stable and unstable manifolds.
    """
    val = 0
    u_manifold = alpha_beta_coords_U(EN, G, Up, H, j, c, d, T)
    u_path_pos = [u_manifold[1][k][0] for k in range(len(u_manifold[1]))]
    u_path_neg = [u_manifold[0][k][0] for k in range(len(u_manifold[0]))]
    s_manifold = alpha_beta_coords_S(EN, G, Up, H, i, a, b, T)
    path = list(nx.shortest_path(EN, i, 0))
    if j not in path:
        return 0
    s_path = list(nx.shortest_path(EN, j, i))
    for k in range(len(u_manifold[0])):
        if u_path_neg[k] in s_path:
            val -= coordinate_intersection(EN, G, H, u_path_neg[k],
                                           s_manifold[1][len(s_path) - 2 - k][0],
                                           s_manifold[1][len(s_path) - 2 - k][1],
                                           u_manifold[0][k][1][0],
                                           u_manifold[0][k][1][1])
        else:
            break
    for k in range(len(u_manifold[1])):
        if u_path_pos[k] in s_path:
            val += coordinate_intersection(EN, G, H, u_path_pos[k],
                                           s_manifold[1][len(s_path) - 2 - k][0],
                                           s_manifold[1][len(s_path) - 2 - k][1],
                                           u_manifold[1][k][1][0],
                                           u_manifold[1][k][1][1])
        else:
            break
    for k in range(len(u_manifold[0])):
        if u_path_neg[k] in s_path:
            val += coordinate_intersection(EN, G, H, u_path_neg[k],
                                           s_manifold[0][len(s_path) - 2 - k][0],
                                           s_manifold[0][len(s_path) - 2 - k][1],
                                           u_manifold[0][k][1][0],
                                           u_manifold[0][k][1][1])
        else:
            break
    for k in range(len(u_manifold[1])):
        if u_path_pos[k] in s_path:
            val -= coordinate_intersection(EN, G, H, u_path_pos[k],
                                           s_manifold[0][len(s_path) - 2 - k][0],
                                           s_manifold[0][len(s_path) - 2 - k][1],
                                           u_manifold[1][k][1][0],
                                           u_manifold[1][k][1][1])
        else:
            break
    return val



def monodromy_matrix(EN, G, Up, H, T):
    """
    Assemble the full monodromy matrix from the blocks associated with each node.
    """
    blocks = []
    ord_nodes = ordered_nodes(EN,G,H)
    for i in ord_nodes:
        greater_neighbors_i = greater_neighbors_EN_sorted(EN,G,Up,H,i)
        if i == 0:
            for a in range(1, len(greater_neighbors_i)):
                blocks.append(monodromy_block_Bia(G, EN, H, i, a))
        elif i != 0 and i not in H.nodes():
            for a in range(len(greater_neighbors_i)):
                blocks.append(monodromy_block_Bia(G, EN, H, i, a))
    temp = []
    for i in range(len(blocks)):
        t_row = []
        for j in range(len(blocks)):
            if i == j:
                t_row.append(blocks[j])
            else:
                t_row.append(np.zeros((blocks[i].shape[0], blocks[j].shape[1]), dtype=int))
        temp.append(t_row)
    M_mat = np.block(temp)

    for i in ord_nodes:
        a_i = EN.nodes[i]['seifert_alpha']
        m_i = G.nodes[i]['multiplicity']
        path = list(nx.shortest_path(EN, i, 0))
        greater_neigh = greater_neighbors_EN_sorted(EN,G,Up,H, i)
        for j in range(1, len(path)):
            greater_neighbors_j = greater_neighbors_EN_sorted(EN,G,Up,H,path[j])
            a_j = EN.nodes[path[j]]['seifert_alpha']
            m_j = G.nodes[path[j]]['multiplicity']
            for a in range(len(greater_neigh)):
                b_range = int(m_i * (1 - Fraction(1, a_i)) + 1) if a == 0 else m_i +1
                for b in range(1, b_range):
                    c = ath_neighbor_inv_EN(EN,G,Up, H, path[j], path[j - 1])
                    if c == 1:
                        if leaf(EN, H, path[j]) != -1:
                            for d in range(1, int(m_j * (1 - Fraction(1, a_j)) + 1)):
                                M_mat[entry_mat(EN, G, Up, H, path[j], c - 1, d)][entry_mat(EN, G,Up, H, i, a, b)] += \
                                intersection_stable_unstable(EN, G, Up, H, i, a, b, path[j], c - 1, d, T)
                    if c > 1:
                        for d in range(1, m_j + 1):
                            M_mat[entry_mat(EN, G, Up, H, path[j], c - 1, d)][entry_mat(EN, G,Up, H, i, a, b)] += \
                            intersection_stable_unstable(EN, G, Up, H, i, a, b, path[j], c - 1, d, T)
                    if c < len(greater_neighbors_j):
                        for d in range(1, m_j + 1):
                            M_mat[entry_mat(EN, G, Up, H, path[j], c, d)][entry_mat(EN, G, Up, H, i, a, b)] += \
                            intersection_stable_unstable(EN, G, Up, H, i, a, b, path[j], c, d, T)
    return np.matrix(M_mat)

def variation_matrix(EN, G, Up, H, T):
    """
    Assemble the full variation matrix.
    """
    n = size_monodromy(EN, G, Up, H)
    V_mat = np.zeros((n,n),dtype=int)
    for p in range(n):
        i, a, b = entry_mat_inv(EN,G,Up,H,p)
        m_i = G.nodes[i]['multiplicity']
        abcup = alpha_beta_coords_U(EN, G, Up, H, i, a, b, T)
        for q in range(n):
            if p == q:
                V_mat[q][p] = 1
            else:
                j, c, d = entry_mat_inv(EN, G, Up, H, q)
                abcuq = alpha_beta_coords_U(EN, G, Up, H, j, c, d, T)
                for t in range(2):
                    for s in range(2):
                        for k in range(len(abcup[t])):
                            for h in range(len(abcuq[s])):
                                val = 0
                                if abcup[t][k][0] == abcuq[s][h][0]:
                                    l = abcup[t][k][0]
                                    a_s = abcup[t][k][1][0]
                                    b_s = abcup[t][k][1][1]
                                    a_u = abcuq[s][h][1][0]
                                    b_u = abcuq[s][h][1][1]
                                    val = ((-1)**(s+t))*coordinate_intersection(EN,
                                                                                G,
                                                                                H,
                                                                                l,
                                                                                a_s,
                                                                                b_s,
                                                                                a_u,
                                                                                b_u,
                                                                                True,
                                                                                1-t)
                                V_mat[q][p] += val
    return np.asmatrix(V_mat)

def differential_map(EN, G, Up, H, T):
    m_0 = G.nodes[0]['multiplicity']
    n = size_monodromy(EN, G, Up, H)
    D = np.zeros((m_0,n), dtype = int)
    for p in range(n):
        i, a, b = entry_mat_inv(EN, G, Up, H, p)
        neg_sign_coord = alpha_beta_coords_S(EN, G, Up, H, i, a, b, T)[0][-1]
        pos_sign_coord = alpha_beta_coords_S(EN, G, Up, H, i, a, b, T)[1][-1]

        D[int(np.floor(neg_sign_coord[0]*m_0)), p] += 1
        D[int(np.floor(pos_sign_coord[0]*m_0)), p] -= 1
    return np.matrix(D)

# =============================================================================
# Main Function to Run the Resolution Computation via Singular
# =============================================================================

def run_polynomial(polynomial):
    """
    Run a given polynomial through Singular to compute the incidence matrix,
    multiplicity matrix, and their inverse. Then build the weighted dual graph,
    compute the splice graph (EN), and assign an ordering to the nodes.

    Returns:
        G_local: The weighted dual graph.
        H_sub: The subgraph corresponding to arrow nodes.
        EN: The spliced graph.
        multiplicity_matrix: The multiplicity matrix from Singular.
    """
    singular_process = start_singular()
    for cmd in start_ring_commands:
        singular_talk(cmd, singular_process)
    singular_talk(f'poly f = {polynomial};', singular_process)

    str_incidence_matrix = singular_talk('proximitymatrix(f)[2];', singular_process)
    incidence_matrix = string_to_numpy_matrix(str_incidence_matrix)

    singular_talk('list tm = totalmultiplicities(f);', singular_process)
    str_multiplicity_matrix = singular_talk('tm[2];', singular_process)
    multiplicity_matrix = string_to_numpy_matrix(str_multiplicity_matrix)

    str_inv_incidence_matrix = singular_talk('print(inverse(proximitymatrix(f)[2]));', singular_process)
    inv_incidence_matrix = string_to_numpy_matrix(str_inv_incidence_matrix)

    total_mult = [sum(row) for row in multiplicity_matrix]
    G_local, H_sub, Up_sub = define_weighted_dual_graph(incidence_matrix, total_mult, inv_incidence_matrix)

    # Compute the splice graph (EN) from the resolution graph and subgraph
    EN = splice_graph(G_local, H_sub)

    # Order the nodes of the resolution graph (side-effect: assigns 'order' attributes)
    ordering(G_local, H_sub, 0, 0)

    singular_process.stdin.write('exit;\n')
    return G_local, H_sub, Up_sub, EN, multiplicity_matrix

# =============================================================================
# Smith normal form (by AI)
# =============================================================================



def swap_rows(A, U, i, j):
    if i != j:
        A.row_swap(i, j)
        U.row_swap(i, j)

def swap_cols(A, V, i, j):
    if i != j:
        A.col_swap(i, j)
        V.col_swap(i, j)

def add_row_multiple(A, U, target, source, k):
    if k:
        A.row_op(target, lambda v, j: v + k * A[source, j])
        U.row_op(target, lambda v, j: v + k * U[source, j])

def add_col_multiple(A, V, target, source, k):
    if k:
        A.col_op(target, lambda v, i: v + k * A[i, source])
        V.col_op(target, lambda v, i: v + k * V[i, source])

def smith_normal_form_custom(matrix):
    A = sp.Matrix(matrix)
    m, n = A.shape
    U = sp.eye(m)
    V = sp.eye(n)
    k = 0

    while k < m and k < n:
        # find smallest nonzero entry in A[k:,k:]
        minval = None
        pos = (k, k)
        for i in range(k, m):
            for j in range(k, n):
                if A[i, j] != 0:
                    a = abs(A[i, j])
                    if minval is None or a < minval:
                        minval, pos = a, (i, j)
        if minval is None:
            break

        i0, j0 = pos
        swap_rows(A, U, k, i0)
        swap_cols(A, V, k, j0)

        if A[k, k] < 0:
            A.row_op(k, lambda v, _: -v)
            U.row_op(k, lambda v, _: -v)

        # Phase 2: Euclidean reduction of pivot
        while True:
            changed = False
            for i in range(m):
                if i != k and A[i, k] != 0:
                    q = A[i, k] // A[k, k]
                    add_row_multiple(A, U, i, k, -q)
                    if A[i, k] != 0:
                        swap_rows(A, U, k, i)
                        if A[k, k] < 0:
                            A.row_op(k, lambda v, _: -v)
                            U.row_op(k, lambda v, _: -v)
                        changed = True
                        break
            if changed:
                continue
            for j in range(n):
                if j != k and A[k, j] != 0:
                    q = A[k, j] // A[k, k]
                    add_col_multiple(A, V, j, k, -q)
                    if A[k, j] != 0:
                        swap_cols(A, V, k, j)
                        if A[k, k] < 0:
                            A.row_op(k, lambda v, _: -v)
                            U.row_op(k, lambda v, _: -v)
                        changed = True
                        break
            if not changed:
                break

        # Phase 3: enforce divisibility in the submatrix
        while True:
            pivot = A[k, k]
            triggered = False
            for i in range(k+1, m):
                for j in range(k+1, n):
                    if A[i, j] % pivot != 0:
                        add_row_multiple(A, U, k, i, 1)
                        triggered = True
                        break
                if triggered:
                    break
            if not triggered:
                break

        k += 1

    # ensure nonnegative pivots
    for i in range(min(m, n)):
        if A[i, i] < 0:
            A.row_op(i, lambda v, _: -v)
            U.row_op(i, lambda v, _: -v)

    return A, U, V

