# Nonconvex GTF

R. Varma, H. Lee, J. Kovačević, and Y. Chi, “Vector-Valued Graph Trend Filtering with Non-Convex Penalties,” in *IEEE Transactions on Signal and Information Processing over Networks*, 2019. **Submitted**.

You can view the paper on arXiv https://arxiv.org/abs/1905.12692.

* To get results in Figure 3 (numbers can be different from random noise)

    middle panel: script_denoising_simulation.py 

    right panel: script_diff_snr_measurements.py

    datasets/2d-grid

    datasets/minnesota


* To get results in Table 1 (numbers can be different from random noise)

	script_UCI_clean_data.py

	script_UCI_classification.py

	script_UCI_get_pvalues.py

	datasets/UCI_data


* To get results in Figure 5 (numbers can be different from random noise)

	support_recovery.ipynb


* All other figures

	plotting.ipynb


* This code was developed in...
	* Python 2.7
	* Numpy 1.11.3
	* Scipy 1.1.0
	* Pandas 0.23.4
	* Matplotlib 2.2.3
	* NetworkX 2.1
	* Hyperopt 0.2
	* PyGSP 0.5.1
	
