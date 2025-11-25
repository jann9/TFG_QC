**Programmers: :shipit:** Mario (University of Malaga), and Zakaria Abdelmoiz DAHI (Inria, University of Lille). 


Two folders:
- **NO_NAS_MODELS:** Uses the Multi-Layer Perceptron (MLP), and the Graphical Neural Networks (GNET) without any Neural Architecture Search (NAS).
- **NAS_MODELS:** Perform Neural Architecture Search (NAS) on the Multi-Layer Perceptron (MLP), and the Graphical Neural Networks (GNET), using P3 and LTGA graybox algorithms.

- **NO_NAS_MODELS**

	- ``Classical_Optimizer.py`` : First attempt at optimizing a Maxcut problem with QAOA. Not used, obsolete.
	- ``Create_data.py``: Creation of MaxCut instances.
	- ``DataSetGen.py``: Creation of datasets (Q_values) given a MaxCut instances dataset 
	- ``DataSetGenCircuit.py``: Same as previous but with circuit level information instead.
	- ``Graph_Neural_Network.py``: Definition and training of graph neural network models
	- ``Model_test.py``: First testing of the models. Not used yet.
	- ``Neural_Models_Train.py``: Definition and training of multilayer perceptron
	- ``Results.py``: Creation of graphs and visuals of the training results
	- ``XGBoost_Model_Train.py``: Definition and training of neural models
	- ``dataset.csv``: MaxCut dataset
	- ``Graph_Neural_Network.py``: which now saves the  graph NN model after training.
	- ``Model_test.py``:  Implement the testing of the trained XGBoost, MLP, and Graph NN models.
	- ``Classical_ML_Comp.py``:  Implement comparison QAOA vs QAOA-based ML
	- ``requirements.txt``: Requirements for installation
	- ``Process_Stats.py``: Process the results of the ``n`` executions
	- ``slurm_launch.sh``: launch ``n`` experiments on slurm cluster


- **NAS_MODELS**
	- **GNET:**
		- src: contains the C++ source code of the NAS algorithms P3 and LTGA.
		- Rlease: Contains the compiled codes of P3 and LTGA, as well as the NO-NAS_MODELS files needed to run the NAS routine on the MLP and GNET models.
	- **MLP:** is organised the same as GNET but for MLP.

- **HOW NAS WORKS**
  - **MLP:** we evolve the sizes of the three hidden layers, as well as the number of iterations. The values evolve between the original value used in the ``NO_NAS_MODELS`` and 1/4 of that value.
  - **GNET:** for the moment we evolve only the size of the hidden layer. As or for the borns of the search, we follow the methodology as done for MLP, where value evolves between the value used in ``NO_NAS_MODELS`` and and its quarter.


- **DETAILS TO BE CLEARED:**
  - What is mean/median, size of graphs or executions.
  - Results of test or training? in MLP looks its is testing, while for G-Net, it is only training.
  - Seems test is on Q-values model only (No circuit model)?
  - The comparison is only for XGBoost and MLP? (see ``model_list = ['xgboost', 'MLP']``) Added GCN to the models and it generates an error.


- **TO DO:**
  - **@Mario:**
    - [ ] **Save graph NN model after training**.
    - [ ] **Implement the testing of the trained XGBoost, MLP, and Graph NN models:** Please, implement something that test all models over all the benchmarks like you did in the training phase. Also, use the same metrics (MAPE, and RMSE) as those used in the training phase. Please, make sure the final results of testing (Mean, Median, Interquartile Range, Standard deviation) are saved in CSV or TXT files.
    - [ ] **Implement comparison QAOA vs QAOA-based ML:** Please, for all the dataset, execute the classical QAOA using some classical optimiser given in Qiskit. Do the same for the proposal QAOA that uses the MLP and graph-based NN to train its parameters. The comparison will be based on the objective function of the MaxCut (Not training metrics). The idea is to compare the efficiency of the proposed QAOA-ML vs Classical QAOA for solving MaxCut.
  - **@Zakaria:**
    - [x] Implement neural architecture search over models.
    - [x] Literature review.
    - [ ]  manuscript writing.


    ![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](/NAS_MODELS/Call_Schema.jpeg)
