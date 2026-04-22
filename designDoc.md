## Using hill climbing search to find optimal tertiary structures of molocules
This is the design document for this research project. What a design document is; I don't know, but this is that. Deal with it.
#### Steps
- Discuss project with Prof. Reza
- Get access to WebMO API
  - Successfully communicate between Python and API
  - Alternatively can use OpenMM simulation
    - Would need to figure out how to display .dcd file
- Create any algorithm and test
- Design storage system for results
- Redesign algorithm to be async so 8 jobs can be run at once
  - Or 4 if using 2 cores
### How it will work
- How data is laid out
  - One atom in each molocule is designated as the root node
  - One molocule in each experiment is designated as the root molocule
  - The root molocule's root node is set as coordinates (0,0,0) and the element
  - Every other atom in the molocule is described using the atom it's connected to, it's element, and the vector the last connected atom
    - Doing it this way allows caps to be set on the bond length and pitch (aka angle)
  - The root node in each other molocule has adjustable coordinates
  - And the atoms in those molocules are described in the same way
- Obviously fitness is just total energy
### Possible questions I'm answering
What search algorithm is best in this context
||Random restart|Beam search|Simulated annealing|Genetic algorithm|Minima hopping method |
|-----------| ----------- | ----------- | ----------- | ----------- | ----------- |
|Prediction| good | same as random restart | ----------- | ----------- | great |
### Articles
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3289079/
- https://web.cs.umass.edu/publication/docs/2004/UM-CS-2004-048.pdf
- https://arxiv.org/pdf/cond-mat/0402136 <- most important
### Notes
- If using WebMO, using .mol files would be best
- Basin hopping