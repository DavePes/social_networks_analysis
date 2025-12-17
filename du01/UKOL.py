import pywikibot as pw
import networkx as nx
import numpy as np
from pprint import pprint


def check_link(source, target):
    suffix1 = "(identifier)"
    suffix2 = "(Identifier)"
    colon = ":"
    if target.endswith(suffix1) or target.endswith(suffix2):
        #print("Ignoring identifiers.")
        return False
    if colon in target:
        #print("Ignoring links to other namespaces.")
        return False
    if source == target:
        #print("Ignoring self-loops.")
        return False
    return True
def get_ego_network(ego, lang='en'):
    '''Return ego network - a directed graph of Wikipedia pages - around the 
    page with the title ego.
    '''
    G = nx.DiGraph()
    G.add_node(ego)
    page = pw.Page(pw.Site('en'), ego)
    ## neighbors of ego
    lp_titles_dict = {}
    for lp in page.linkedPages():
        lp_titles_dict[lp.title()] = lp
        if check_link(ego, lp.title()):
            G.add_edge(ego, lp.title())
    for lp in page.getReferences():
        lp_titles_dict[lp.title()] = lp
        if check_link(ego, lp.title()):
            G.add_edge(lp.title(), ego)
    ## iterate through neighbors
    for i,node in enumerate(G.nodes()):
        print(f"Processing node {i+1}/{len(G.nodes())}: {node}")
        if (node == ego):
            continue
        page = lp_titles_dict[node]
        for lp in page.linkedPages():
            if lp.title() in lp_titles_dict:
                if check_link(ego, lp.title()):
                    G.add_edge(node,lp.title())

    return G


from math import isclose

ego = 'Bembidion ambiguum'
e = get_ego_network(ego)

print("Number of nodes", len(e))
print("Number of edges", len(list(e.edges())))
assert (isclose(len(e), 21, abs_tol=1) and isclose(len(list(e.edges())), 106, rel_tol=0.02)) #or \
        #(isclose(len(e), 23, abs_tol=1) and isclose(len(list(e.edges())), 133, rel_tol=0.02))
