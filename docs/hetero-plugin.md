# Inference Engine hetero plugin design overview {#openvino_docs_hetero_plugin}

## Subgraphs selection

Algorithm:

For each plugin
1. Select *root* node
    * Node not in subgraph previously constructed
    * Affinity is equal to plugin name
2. Select adjacent node to any node in already subgraph which is not in rejected list
    * if there are no such nodes **end**
3. Check selected node has same affinity
4. Add node to subgraph if check was successful or add to rejected list otherwise
5. Check global condition
    * Nodes in rejected list can never be added to subgraph
    * Nodes not in subgraph and not in rejected list can possibly be added later
    * Check subgraph topology (the only check now is there are no indirect subgraph self-references)
6. If global condition was failed remove last node from subgraph, add it to rejected list and go to step 5
    * we can rollback multiple times here because rejected list is changed every time
7. Go to step 2

Example:
```
    1
    |
    2
   / \
  3   4
   \ /
    5
    |
    6
    |
    7
```

Nodes [1,2,3,5,6,7] are supported in plugin, [4] is not

Possible roots: [1,2,3,5,6,7]
1. Select root [1]
    * Subgraph: [1]
    * Rejected: []
    * Global condition: ok
2. Merge [2]
    * Subgraph: [1,2]
    * Rejected: []
    * Global condition: ok
3. Merge [3]
    * Subgraph: [1,2,3]
    * Rejected: []
    * Global condition: ok
4. Merge [5]
    * Subgraph: [1,2,3,5]
    * Rejected: []
    * Global condition: There is possible self-references through node [4] but we do not know yet, ok
5. Merge [6]
    * Subgraph: [1,2,3,5,6]
    * Rejected: []
    * Global condition: There is possible self-references through node [4] but we do not know yet, ok
6. Merge [7]
    * Subgraph: [1,2,3,5,6,7]
    * Rejected: []
    * Global condition: There is possible self-references through node [4] but we do not know yet, ok
7. Failed to merge [4]
    * Subgraph: [1,2,3,5,6,7]
    * Rejected: [4]
    * Global condition: There is self-references through node [4], reject
8. Rollback [7]
    * Subgraph: [1,2,3,5,6]
    * Rejected: [4,7]
    * Global condition: There is self-references through node [4], reject
9. Rollback [6]
    * Subgraph: [1,2,3,5]
    * Rejected: [4,6,7]
    * Global condition: There is self-references through node [4], reject
10. Rollback [5]
    * Subgraph: [1,2,3]
    * Rejected: [4,5,6,7]
    * Global condition: ok
11. There are nodes to merge **end**

Possible roots: [5,6,7]
1. Select root [5]
    * Subgraph: [5]
    * Rejected: []
    * Global condition: ok
2. Merge [6]
    * Subgraph: [5,6]
    * Rejected: []
    * Global condition: ok
3. Merge [7]
    * Subgraph: [5,6,7]
    * Rejected: []
    * Global condition: ok
4. Merge [3]
    * Subgraph: [3,5,6,7]
    * Rejected: []
    * Global condition: ok
5. Merge [2]
    * Subgraph: [2,3,5,6,7]
    * Rejected: []
    * Global condition: There is possible self-references through node [4] but we do not know yet, ok
6. Failed to merge [4]
    * Subgraph: [2,3,5,6,7]
    * Rejected: [4]
    * Global condition: There is self-references through node [4], reject
7. Rollback [2]
    * Subgraph: [3,5,6,7]
    * Rejected: [2,4]
    * Global condition: ok
8. There are nodes to merge **end**

Possible roots: [] no roots, **END**

Subgraphs: [1,2,3], [3,5,6,7]

Select best subgraph:
* When we have multiple subgraphs larger ([3,5,6,7]) is always selected, always

Repeat previous steps with remaining nodes [1,2]

The final result is:
* First plugin: [3,5,6,7], [1,2]
* Second plugin: [4]


## Subgraphs self reference detection

1. For each node in network build a list of reachable node (transitive closure)
2. For each pair of nodes in subgraph find `path` nodes (nodes through one node in pair reachable to other)
    * assume `src` - one node in pair, `dst` - other node in pair
    * get all nodes reachable from `src`
    * in those nodes find nodes through you can reach `dst` those will be our `path` node
3. Results for pairs is cached.
4. Check if there intersection between `path` nodes set and rejected nodes set for each nodes pair in subgraph
5. In case of intersection we have a self-reference and subgraph is invalid
