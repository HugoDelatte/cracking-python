# Breadth-first search (BFS)

# It is an algorithm for searching a tree for a node that satisfies a given property.
# It starts at the tree root and explores all nodes at the present depth prior to moving on to the nodes at the next
# depth level. Extra memory (like a queue) is needed to keep track of the child nodes encountered and not yet explored.

# Worst case time complexity = O(|V|+|E|) = O(b^d)
# Worst case space complexity = O(|V|) = O(b^d)
# |V|: number of vertices
# |E|: the number of edges
# d: distance  fro the start node measured in number of edge traversals
# b: branching factor of the graph (the average out-degree)

# BFS for a graph is similar to BFS of a tree.
# The only catch here is, unlike trees, graphs may contain cycles, so we may come to the same node again.
# To avoid processing a node more than once, we use a boolean visited array or a visited set.

from collections import deque


# ============================================ Representation of a Graph ===============================================
# An adjacency list is a collection of unordered lists used to represent a finite graph.
# Each unordered list within an adjacency list describes the set of neighbors of a particular vertex in the graph.
# An implementation uses a hash table to associate each vertex in a graph with an array of adjacent vertices.


# ======================================================================================================================
# Simple using set and queue

def bfs_v1(graph: dict, node: str):
    # Checking if an item is in the set in O(1)
    visited = set()
    #  Queue is preferred over list in the cases where we need to add an element on at the start
    # or remove at the end --> O(1) time complexity compared to O(n) for list
    queue = deque()

    visited.add(node)
    queue.append(node)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)


def test_v1():
    graph = {
        '5': ['3', '7'],
        '3': ['2', '4'],
        '7': ['8'],
        '2': [],
        '4': ['8'],
        '8': []
    }
    bfs_v1(graph=graph, node='5')


# ======================================================================================================================
# Using Graph class


class Graph:
    def __init__(self, edges: list[tuple], n: int):
        # A list of lists to represent an adjacency list
        self.adj_list = [[] for _ in range(n)]

        # add edges to the undirected graph
        for (vertex, neighbor) in edges:
            self.adj_list[vertex].append(neighbor)
            self.adj_list[neighbor].append(vertex)


def bfs_v2(graph: Graph, vertex: int, visited: list):
    queue = deque()
    visited[vertex] = True
    queue.append(vertex)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        for neighbor in graph.adj_list[vertex]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True


def test_v2():
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (5, 9), (5, 10), (4, 7), (4, 8), (7, 11), (7, 12)]
    # total number of nodes in the graph (labelled from 0 to 14)
    n = 15
    visited = [False] * n
    # build a graph from the given edges
    graph = Graph(edges, n)
    print(graph.adj_list)
    # Perform BFS traversal from all undiscovered nodes to
    # cover all connected components of a graph
    for i in range(n):
        if not visited[i]:
            # start BFS traversal from vertex i
            bfs_v2(graph, vertex=i, visited=visited)
