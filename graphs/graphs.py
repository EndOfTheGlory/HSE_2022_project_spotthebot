import numpy as np
import pandas as pd
import copy
import scipy
import intertools

# tracking progress
from tqdm import tqdm

# to build graphs
import networkx as nx

# Basic classe to build graphs: Node and Graph
class Node:
    def __init__(self, word: str, word_vect: np.ndarray, neighbors: set):
        # WE ARE CREATING COPIES!!! NOT LINKS TO BETWEEN TARGET AND OBJECT!!
        # https://docs.python.org/3/library/copy.html
        self.__word = copy.deepcopy(word)
        self.__word_vect = copy.deepcopy(word_vect)
        self.__neighbors = copy.deepcopy(neighbors)

    # caller functions(getting LINK to the object)
    @property
    def word(self):
        return self.__word

    @property
    def word_vect(self):
        return self.__word_vect

    @property
    def neighbors(self):
        return self.__neighbors

    # ensuring data encapsulation

    @word.setter
    def word(self, word: str):
        self.__word = copy.deepcopy(word)

    @word_vect.setter
    def word_vect(self, word_vect: np.ndarray):
        self.__word_vect = copy.deepcopy(word_vect)

    @neighbors.setter
    def neighbors(self, neighbors: set):
        self.__neighbors = copy.deepcopy(neighbors)


class Graph:
    def __init__(self, vertices: dict):
        '''
        vertices look like: {word: Node(word_vector, neighbours(set)}
        '''
        self.vertices = copy.deepcopy(vertices)

    @property
    def vertices(self):
        return self.vertices

    @vertices.setter
    def vertices(self, vertices: dict):
        self.vertices = copy.deepcopy(vertices)

    def get_wordvect(self):
        wordvect = list()
        for one_node in self.vertices.values():
            wordvect.append(one_node.word_vector.tolist())
        return wordvect

    def get_words(self):
        return list(self.vertices.keys())

    def getting_all_edges(self):
        edges = set()
        for word, node in self.vertices.items():
            for neighbor in node.neighbors:
                # checking if we already have edge
                if (word, neighbor) in edges or (neighbor, word) in edges:
                    continue
                edges.add((word, neighbor))
        return edges

    def add_edge(self, word_one: str, word_two: str):
        self.vertices[word_one].neighbors.add(second_word)
        self.vertices[word_two].neighbors.add(first_word)

    def delete_edge(self, word_one: str, word_two: str):
        self.vertices[word_one].neighbors.difference_update(set([word_two]))
        self.vertices[word_two].neighbors.difference_update(set([word_one]))

    def reset_graph_neigh(self):
        for one_node in self.vertices.values():
            one_node.neighbors.clear()


    # characteristics of graph(euclid distance, etc.) to  use our graphs

    def euclid_dist(self, word: str, vect: np.ndarray):
        words_vector = self.vertices[word].vector
        return np.linalg.norm(words_vector - vect)

    def euclid_dist_usual(self, first: str, second: str):
        vertex_one = self.vertices[first].vector
        vertex_two = self.vertices[second].vector
        return np.linalg.norm(vertex_one - vertex_two)


    def other_words(self, *words):
        return set(self.get_words()).difference(set(words))

    def nearest(self, word: str):
        your_best_one = ''
        neigh_dist = float('inf')
        for other in self.other_words(word):
            if self.euclid_dist_usual(word, other) < neigh_dist:
                neigh_dist = self.euclid_dist_usual(word, other)
                your_best_one = other
        return your_best_one



    # Needed to build KNN graph
    def knn(self, word: str, k_size: int):
        # doing the same as nearest, just sorting in diminishing order
        dist_to_compare = dict()
        for other in self.other_words(word):
            dist_to_compare[other] = self.euclid_dist_usual(word, other)
        sort_way = lambda x: x[1]
        sorted_words = [key for key, value in sorted(dist_to_compare.items(), key=sort_way)]
        return sorted_words[:k_size]

    # using for IG
    def sphere_radius(self, word: str):
        nearest_negih = self.nearest(word)
        return self.euclid_dist_usual(word, nearest_negih)

    # Needed for building graphs
    # def change_to_networkx(self):
    #     networkx = nx.Graph()


    # One of many characteristics we use to compare graphs


# Functions to fill the graph

def creating_vertices(list_of_articles: list, length_of_vector: int):
    '''
    We have articles, splitted by text, we need to get a vertex out of it

    :param list_of_articles: what we got from vectorizing(list of dictionaries, describing articles-words)
    :param length_of_vector: self-explanatory
    :return: dictionary of the look: {word: Node(word_vect, empty_set)
    '''
    vert_dict = dict()
    file_name_tqdm = open("tracking_result.txt", 'w')

    for one_article in tqdm(list_of_articles, file_name_tqdm):
        for sent in one_article:
            # we are accesing dictionary info
            for word, word_vect in sent['sentence_text'].items():
                # checking if word is already listed
                if word not in vert_dict.keys():
                    try:
                        vert_dict[word] = Node(word=word, word_vect=word_vect, neighbors=set())
                    except:
                        pass

    file_name_tqdm.close()
    return vert_dict

'''
GG & DT - граф Габриэля и Триангуляция Делоне соотвественно
'''

class GG(Graph):
    # if we have Delaunay graph, we can build Gabrielle by O(n)
    def __init__(self, vertices: dict, triang: list=list(),
                 delaunay: scipy.spatial.qhull.Delaunay = None):
        super().__init__(vertices)
        self.triang = copy.deepcopy(triang)
        self.delaunay = copy.deepcopy(delaunay)

    def delaunay(self):
        return self.delaunay

    def triang(self):
        return self.triang

    def create_delaunay(self):
        self.reset_graph_neigh()
        word = self.get_words()
        word_vect = self.get_wordvect()
        file_name_tqdm = open("Delaunay_results.txt", 'w')
        self.delaunay = Delaunay(np.array(vectors))
        delaunay_graph = self.delaunay.simplices.tolist()
        word_num_dict = {word: num for word, num in enumerate(words)}
        for triang in tqdm(delaunay_graph, file_name_tqdm):
            # we are getting triangles, which are included
            triangle_words = set(map(word_num_dict.get, triangle))
            self.triang.append(triangle_words)
            for one_word in triangle_words:
                new_neigh = triangle_words.difference(set([one_word]))
                # we should create setter for this, so that we don't use general Graph(alway updating)
                self.vertices[one_word].neighbors.update(new_neigh)
        file_name_tqdm.close()


    # we need the function to check when delete edges(according to defention of graph)
    def neigh_in_edge_sphere(self, edge: tuple, neigh: str):
        center = (self.vertices[edge[0]].word_vect + self.vertices[edge[1]].word_vect) / 2
        radius = self.euclid_dist(edge[0], center)
        if self.euclid_dist(neigh, center) >= radius:
            return False
        return True

    # creating Gabrielle using Delaunay
    def create_gabrielle_graph(self):
        self.create_delaunay()
        file_name_tqdm = open("Gabrielle_results.txt", 'w')
        for triang in tqdm(self.triang, file_name_tqdm):
            edges = list(intertools.combinations(triag, 2))
            for edge in edges:
                neighs = triag.difference(set(edge))
                for neigh in neighs:
                    # according to definition
                    if self.neigh_in_edge_sphere(edge, neigh):
                        self.delete_edge(edge[0], edge[1])

        file_name_tqdm.close()

    def create_word_graph_gabrielle(self, words: set):
        self.create_delaunay()
        file_name_tqdm = open("Gabrielle_results.txt", 'w')
        for word in tqdm(words, file_name_tqdm):
            other_verteces = self.other_words(word)
            for try_neigh in other_verteces:
                flag = True
                others = self.other_words(word, try_neigh)
                for other in others:
                    try_edge = (word, try_neigh)
                    if self.neigh_in_edge_sphere(try_edge, other):
                        flag = False
                        break
                if flag:
                    self.vertices[word].neighbors.update(try_neigh)
        file_name_tqdm.write("Cycle done")
        ver_with_neighs = dict()
        for word in words:
            ver_with_neighs[word] = self.vertices[word]
        file_name_tqdm.close()
        return ver_with_neighs


'''Influence Graph, IG - Граф влияния'''

class IG(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)

    def create_IG(self):
        self.reset_graph_neigh()
        for first_word_edge in tqdm(self.vertices.keys()):
            others = self.other_words(first_word_edge)
            radius_sphere_for_first = self.sphere_radius(first_word_edge)
            for second_word_edge in others:
                radius_sphere_for_second = self.sphere_radius(second_word_edge)
                distance_between_the_two = self.euclid_dist_usual(first_word_edge, second_word_edge)
                if distance_between_the_two <= radius_sphere_for_second + radius_sphere_for_first:
                    self.add_edge(second_word_edge, first_word_edge)

    def create_word_graph(self, words: set, sphere_radius: int):
        self.reset_graph_neigh()
        file_name_tqdm = open("IG_words_results.txt", 'w')
        for first_word_edge in tqdm(words, file = file_name_tqdm):
            others = self.other_words(first_word_edge)
            radius_sphere_for_first = self.sphere_radius(first_word_edge)
            for second_word_edge in others:
                radius_sphere_for_second = self.sphere_radius(second_word_edge)
                distance_between_the_two = self.euclid_dist_usual(first_word_edge, second_word_edge)
                if distance_between_the_two <= radius_sphere_for_second + radius_sphere_for_first:
                    self.add_edge(second_word_edge, first_word_edge)
        file_name_tqdm.write("Cycle done")
        ver_with_neighs = dict()
        for word in words:
            ver_with_neighs[word] = self.vertices[word]
        file_name_tqdm.close()
        return ver_with_neighs



'''Nearest Neighbours Graph(NNG) - Граф k-ближайших соседей'''

class NNG(Graph):
    def __init__(self, vertices: dist):
        super().__init__(vertices)

    # every vertex is connected with it's k nearest neigbhours
    def create_nng(self, num_neighs_k: int):
        self.reset_graph_neigh()
        file_name_tqdm = open("IG_result.txt", 'w')

        for word in tqdm(self.vertices.keys(), file=file_name_tqdm):
            list_for_graph = self.knn(word, num_neighs_k)
            for one_neigh in list_for_graph:
                self.add_edge(word, one_neigh)
        file_name_tqdm.write("Cycle done")
        ver_with_neighs = dict()
        for word in words:
            ver_with_neighs[word] = self.vertices[word]
        file_name_tqdm.close()
        return ver_with_neighs


'''e-ball - граф e(эпсилон)-окружности'''

class E_Ball(Graph):
    def __init__(self, vertices: dict):
        super().__init__(vertices)

    def creating_eball(self, eps: float):
        self.reset_graph_neigh()
        for vertex_one in self.get_words():
            others = self.other_words(vertex_one)
            for second_vertex in others:
                dist_to_comp = self.euclid_dist_usual(vertex_one, second_vertex)
                if dist_to_comp < eps:
                    self.add_edge(vertex_one, second_vertex)


    # to test out(look at the part of the graph)
    def creating_word_graph(self, words: set, eps: float):
        self.reset_graph_neigh()
        file_name_tqdm = open("eball_result.txt", 'w')
        for vertex_one in tqdm(words, file_name_tqdm):
            others = self.other_words(vertex_one)
            for second_vertex in others:
                dist_to_comp = self.euclid_dist_usual(vertex_one, second_vertex)
                if dist_to_comp < eps:
                    self.add_edge(vertex_one, second_vertex)
        file_name_tqdm.write("Eball cicle ended")
        ver_with_neighs = dict()
        for word in words:
            ver_with_neighs[word] = self.vertices[word]
        file_name_tqdm.close()
        return ver_with_neighs

'''Relative Neighborhood Graph(RNG) - Граф относительного соседства'''


# it's subgraph of Gabrielle's graph, if we have Gabrielle, we can count by O(n^2)
class RNG(GG):
    def __init__(self, vertices: dict, triag: list = list(), delaunay: scipy.spatial.qhull.Delaunay = None):
        super().__init__(vertices, triag, delaunay)

    def create_rng(self, is_already_gabrielle: bool):
        if not is_already_gabrielle:
            self.create_gabrielle_graph()
        file_name_tqdm = open('rng_result', 'w')
        for word in tqdm(self.get_words(), file=file_name_tqdm):
            neighs = copy.deepcopy(self.vertices[word].neighbors)
            for neigh in neighs:
                # we can accept multiple values to delete
                others = self.other_words(word, neigh)
                for other in others:
                    word_neigh_dist = self.euclid_dist_usual(word, neigh)
                    neigh_other_dist = self.euclid_dist_usual(neigh, other)
                    word_other_dist = self.euclid_dist_usual(word, other)
                    if max(word_other_dist, word_neigh_dist) > word_neigh_dist:
                        continue
                    self.delete_edge(word, neigh)
        file_name_tqdm.close()

        def create_word_rng(self, words: set, is_already_gabrielle: bool):
            if not is_already_gabrielle:
                self.create_gabrielle_graph()
            file_name_tqdm = open('rng_words_result', 'w')
            for word in tqdm(words, file=file_name_tqdm):
                neighs = copy.deepcopy(self.vertices[word].neighbors)
                for neigh in neighs:
                    # we can accept multiple values to delete
                    others = self.other_words(word, neigh)
                    for other in others:
                        word_neigh_dist = self.euclid_dist_usual(word, neigh)
                        neigh_other_dist = self.euclid_dist_usual(neigh, other)
                        word_other_dist = self.euclid_dist_usual(word, other)
                        if max(word_other_dist, word_neigh_dist) > word_neigh_dist:
                            continue
                        self.delete_edge(word, neigh)
            file_name_tqdm.write("RNG cycle ended")
            ver_with_neighs = dict()
            for word in words:
                ver_with_neighs[word] = self.vertices[word]
            file_name_tqdm.close()
            return ver_with_neighs