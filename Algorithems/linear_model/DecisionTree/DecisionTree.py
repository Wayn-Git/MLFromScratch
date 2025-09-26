"""
How decision tree actually works ? -> 

Decision tree is a supervised machine learning algorithems that serves it's purpose in both 
regression as well as classification problems. 

The structure includes a root node (Starting point), branches (Represents choices and outcomes) and the leaf nodes (The final decision or prediction)


Key Components: Root Node -> The first node in the tree representing the entire dataset 
internal Nodes -> Nodes where a test or a condition is applied to a feature of the data

Branches -> Connections between nodes that represent the outcome

Leaf Node (Aka Terminal Nodes) -> The end node of the tree with no further children node or bracnhes, it contains the final prediction of the decision tree

How it works -> 

1. Starting Point (Root node) -> The process begins with the root node with entire dataset

2. Splitting -> At each internal node, the data is split based on the specific feature and condition

3. Branching -> These splits lead to different bracnhes that carry down the data based on the test outcome to other nodes

4. Conclusion: The process continues until we reach the leaf node


Key Points to build from scratch ->

Entropy ( a concept that measures randomness, uncertainty with a system, we can use this is measure how sure the tree is about it's prediction when moving to the next node)

"""