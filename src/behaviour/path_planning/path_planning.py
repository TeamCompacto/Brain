
from path_finding_algorithm.entities import Edge
from path_finding_algorithm.entities import Node
from path_finding_algorithm.a_star import AStar
from path_finding_algorithm.read_information import Reader

if __name__ == '__main__':
    g = Reader('Test.graphml').G #The parameter is the name of the file containing the road map represented by a directed graph
    nodes = {int(node): Node(int(node), data['x'], data['y']) for node, data in g.nodes(data=True)}
    edges = [Edge(int(src), int(dest), data['dotted']) for src, dest, data in g.edges(data=True)]
    
    a_star = AStar(nodes, edges, 2, 3)

    
