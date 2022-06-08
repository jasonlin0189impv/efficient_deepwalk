# Efficient DeepWalk
Use efficient random walk algorithm to reduce computation time in deepwalk

# Simple experiments

## Dataset info
| Dataset   | Number of Nodes  | Number of Edges  | Number of Classes |
| --------- |-----------------:| ----------------:| -----------------:|
| Cora      | 2708             | 5278             | 7                 |
| PubMed    | 19717            | 44324            | 3                 |


## Experiments (use MacBook M1 pro)
* Setting
| Dataset  | Walk Length | Number of Walks | Embedding Size | Window Size |
| -------: |------------:| ---------------:| --------------:|------------:|
| Cora     | 20          | 30              | 64             | 5           |
| PubMed   | 20          | 20              | 64             | 5           |

* Experiments (test size 0.3)
|Method      | Dataset  | Computing Time | Training F1 (micro) | Testing F1 (micro) |
|----------- | -------: |---------------:| -------------------:| ------------------:|
|Basic RW    | Cora     | 13.8 s         | 0.393               | 0.299              |
|Efficient RW| Cora     | 2.5 s          | 0.395               | 0.299              |
|Basic RW    | PubMed   | 133 s          | 0.550               | 0.399              |
|Efficient RW| PubMed   | 28.3 s         | 0.543               | 0.400              |

## Conclusion
* Efficient random walk spend less time but still have similar performance with basic random walk.
