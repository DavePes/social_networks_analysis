class GraphAdjMat:
    """ Graph implemented as adjacency list stored as a dictionary"""
    
    def __init__(self):
        self.adj_matrix = {}            # a dictionary of adjacency dictionaries
        self.vertices = {}              # a dictionary of vertices
    
    def add_vertex(self, id, data=None):
        """Add one vertex with key id into the graph
        if the node with id is already in the graph, then do nothing, 
        otherwise, add it to the self_adj_matrix with an empty list (dictionary) 
        of neighbors
        If data is nonempty, store it in self.vertices dictionary
        """
        if (id not in self.adj_matrix):
            self.adj_matrix[id] = {}
            if data is not None:
                self.vertices[id] = data
            
        
    def add_edge(self, u_id, v_id, data:dict=None):
        """Add one edge from the vertex with u_id to the vertex v_id
        The edge will be stored in the adjacency dictionary together with the data.
        The data should be a dictionary.
        """
        if u_id in self.adj_matrix:
            self.adj_matrix[u_id][v_id] = data
        else:
            self.adj_matrix[u_id] = {v_id: data}


    def print_vertices(self):
        """Print list of nodes together with their data in the format
        (node1_id, node1_data), (node2_id, node2_data), ...
        """
        l = []
        for id, data in self.vertices.items():
            l.append(f"({id}, {data})")

        print(', '.join(l))
    
    def print_graph(self):
        """Print adjacency matrix together with node names.
        each line should be of the form
        
            id(data) : (n1: d1), (n2: d2) (n3: d3) ...
        
        where 
            id is the id of a node,            
            data is the data of the node,
            n1 is the id of the first neighbor of the node id, 
            d1 is the data associated with edge (id,w1), 
            n2 is the id of the second neighbor of the node id, etc.
        Be aware that id, n1, n2, ... are any hashable values!
        """
        for id, data in self.vertices.items():
            print(f"{id}({data}) : ", end='')

# g = GraphAdjMat()
# g.add_vertex('A', 'node A data')
# g.add_vertex('B', 'node B data')
# g.add_edge('A', 'B')
# g.print_graph()

a = {'key1': {}, 'key2': {}}

print(a['key1'])
a['key2']['di2'] = 5
print(a['key2']['di2'])