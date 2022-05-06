<!-- #region -->
# Nonconvex GTF

R. Varma*, H. Lee*, J. Kovačević and Y. Chi, “Vector-Valued Graph Trend Filtering With Non-Convex Penalties,” in IEEE Transactions on Signal and Information Processing over Networks, vol. 6, pp. 48-62, 2020. 

https://ieeexplore.ieee.org/document/8926407

You can view the paper on arXiv https://arxiv.org/abs/1905.12692.

Note that when you run these scripts, numbers can be different from what is reported in the paper due to random noise.

	* Figure 4 
	
	script/support_recovery.ipynb
    
    
	* Figure 5
	
	middle panel: script(3)/script_denoising_simulation.py
	
	right panel: script3/Runtime experiment.ipynb
	
	datasets/2d-grid
	
	datasets/minnesota
	
	
	* Table 3
	
	script(3)/script_diff_snr_measurements.py


	* Table 4
	
	script(3)/script_UCI_clean_data.py
	
	script(3)/script_UCI_classification.py
	
	script(3)/script_UCI_get_pvalues.py
	
	datasets/UCI_data
    
    
	* Figure 6
	
	script3/NYC_taxi.ipynb
    
    
	* All other figures
	
	script/plotting.ipynb
<!-- #endregion -->

GTF/ and scripts/ are developed in Python 2.7, and
	
	* Numpy 1.11.3
	
	* Scipy 1.1.0
	
	* Pandas 0.23.4
	
	* Matplotlib 2.2.3
	
	* NetworkX 2.1
	
	* Hyperopt 0.2
	
	* PyGSP 0.5.1
    
GTF3/ and scripts3/ are developed in Python 3.6, and
	
	* Numpy 1.17.0
	
	* Scipy 1.1.0
	
	* Pandas 0.24.2
	
	* Matplotlib 3.0.1
	
	* NetworkX 2.3
	
	* Hyperopt 0.2
	
	* PyGSP 0.5.1
    	
	* tqdm 4.26.0
