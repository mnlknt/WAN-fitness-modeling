# WAN-fitness-modeling
Minimal codes for all reconstruction methods used in G. Fischetti, A. Mancini, G. Cimini, A. Vespignani and G. Caldarelli, Fitness modeling of the Worldwide Air Transportation Network for epidemic forecasting (2025).

- UNDIR contains all codes for the reconstruction of undirected networks.
- DIR contains all codes for the reconstruction of directed networks.

Each folder contains the following codes:

- rec_functions.py, where all functions used for the reconstruction are defined
- bdgm_functions.py, where all fuctions for the reconstruction using the block degree-corrected gravity model are defined
- reconstruction_(un)directed.py, where network and reconstruction parameters need to be defined and the corresponding functions are called

'input_path' must be a path to a folder containing all files in the (UN)DIR/input folder.

Mapping of models names used in codes to model names used in article:
- dgm : model K
- nlin-dgm : model S
- dist-dgm : model D
- bdgm : model B
- comb-bdgm : model C
