import networkx as nx
import matplotlib.pyplot as plt


class Reader():
    def __init__(self, file_name) -> None:
        self.__G = nx.read_graphml(file_name)
        # nx.draw(self.G)
        # plt.show()

    @property
    def G(self):
        return self.__G