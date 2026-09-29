import networkx as nx
import pyvis
from pyvis.network import Network
from graphviz import Digraph

from .engine import Value

def show_graph_interactive(self, filename="graph.html"):
    nodes = {} # Diccionario para almacenar nodos por ID
    edges = []

    def build(v):
        if id(v) not in nodes: #Usamos el id del objeto como key
            nodes[id(v)] = {"label": f"{v.name} | data={v.data:.2f} | grad={v.grad:.2f}", "shape": "box"}
            if v._op: # Si tiene una operación, crea el nodo de operación
                op_node_id = f"op_{id(v)}" #ID unico para el nodo de operacion
                nodes[op_node_id] = {"label": v._op, "shape": "circle", "color": "lightblue", "size": 20} #Nodo de operacion
                for child in v._prev:
                    edges.append((id(child), op_node_id)) #Arista desde los hijos a la operacion
                edges.append((op_node_id, id(v))) #Arista desde la operacion al resultado
                for child in v._prev:
                    build(child) #Llamada recursiva para los hijos

    build(self)

    graph = nx.DiGraph()
    for node_id, node_data in nodes.items():
        graph.add_node(node_id, **node_data) #Agrega los nodos al grafo con sus atributos
    for edge in edges:
        graph.add_edge(edge[0], edge[1], arrows={'to': True, 'from': False})

    net = Network(notebook=True, cdn_resources='in_line', directed=True)

    net.from_nx(graph)
    net.prep_notebook()
    net.show(filename)

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def show_graph(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{%s | data %.4f | grad %.4f }" % (n.name, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

