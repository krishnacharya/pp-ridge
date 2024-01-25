# pp-ridge
Personalized Privacy for Ridge Regression

Abstract

```
The increased application of machine learning (ML) in sensitive domains requires protecting the training data through privacy frameworks, such as differential privacy (DP). DP requires to specify a uniform privacy level ε that expresses the maximum privacy loss that each data point in the entire dataset is willing to tolerate. Yet, in practice, different data points often have different privacy requirements. Having to set one uniform privacy level is usually too restrictive, often forcing a learner to guarantee the stringent privacy requirement, at a large cost to accuracy. To overcome this limitation, we introduce our novel PersonalizedDP Output Perturbation method (PDP-OP) that enables to train Ridge regression models with individual per data point privacy levels. We provide rigorous privacy proofs for our PDP-OP as well as accuracy guarantees for the resulting model. This work is the first to provide such theoretical accuracy guarantees when it comes to personalized DP in machine learning, whereas previous work only provided empirical evaluations. We empirically evaluate PDP-OP on synthetic and real datasets and with diverse privacy distributions. We show that by enabling each data point to specify their own privacy requirement, we can significantly improve the privacy-accuracy trade-offs in DP. We also show that PDP-OP outperforms the personalized privacy techniques of Jorgensen et al. (2015).
```


```bash
datasets
   |-- insurance.csv
notebooks
   |-- PP-on-realdata.ipynb
   |-- PP-on-synthetic.ipynb
plots
   |-- .DS_Store
   |-- synthetic_sigma0
   |   |-- .DS_Store
   |   |-- impact_epsc
   |   |   |-- impact_epsc_sec4_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsc_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsc_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsc_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsc_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |-- impact_epsm
   |   |   |-- impact_epsm_sec4_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsm_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsm_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |   |-- impact_epsm_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_epsm_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100_lambda_50.pdf
   |   |-- impact_fc
   |   |   |-- .DS_Store
   |   |   |-- impact_fc_sec4_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100_lambda_100.pdf
   |   |   |-- impact_fc_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100_lambda_100.pdf
   |   |   |-- impact_fc_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100_lambda_100.pdf
   |   |   |-- impact_fc_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- impact_fc_sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100_lambda_100.pdf
   |   |-- sanity_check
   |   |   |-- nonpriv_soln_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |-- sec_4_1
   |   |   |-- .DS_Store
   |   |   |-- sec4_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_1_type1_syn_plevel_34_43_23_d_30_n_100_p2.pdf
   |   |   |-- sec4_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_1_type2_syn_plevel_34_43_23_d_30_n_100_p2.pdf
   |   |-- sec_4_2_1
   |   |   |-- sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_2_1_type1_syn_plevel_34_43_23_d_30_n_100_p2.pdf
   |   |   |-- sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_2_1_type2_syn_plevel_34_43_23_d_30_n_100_p2.pdf
   |   |-- sec_4_2_2
   |   |   |-- .DS_Store
   |   |   |-- sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_2_2_type1_syn_plevel_34_43_23_d_30_n_100_p2.pdf
   |   |   |-- sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100.pdf
   |   |   |-- sec4_2_2_type2_syn_plevel_34_43_23_d_30_n_100_p2.pdf
pp
   |-- .DS_Store
   |-- check.py
   |-- csv_outputs
   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23_affine.csv
   |   |-- forplots_insurance_data_impact_epsc_epsm_affine.csv
   |   |-- forplots_insurance_data_impact_fc_affine.csv
   |   |-- forplots_insurance_data_impact_lambda_plevel_34_43_23_affine.csv
   |   |-- forplots_insurance_data_impact_n_plevel_34_43_23_affine.csv
   |   |-- forplots_insurance_data_impact_n_plevel_54_37_9_affine.csv
   |   |-- forplots_specific_30_100_plevel_34_43_23_result.csv
   |   |-- forplots_specific_30_100_plevel_54_37_9_result.csv
   |   |-- insurance_data_plevel344323_affine_lambdasbig.csv
   |   |-- insurance_data_plevel344323_affine_lotslambdas_runs500.csv
   |   |-- reg_specific_30_100_plevel_34_43_23_result.csv
   |   |-- reg_specific_30_100_plevel_54_37_9_result.csv
   |   |-- unreg_specific_30_100_plevel_34_43_23_result.csv
   |   |-- unreg_specific_30_100_plevel_54_37_9_result.csv
   |-- interpret_and_plots
   |   |-- Interpret_results_insurance_affine.ipynb
   |   |-- Interpret_results_synthetic.ipynb
   |-- plevel_34_43_23_result_findlambs.csv
   |-- real_data
   |   |-- california_housing
   |   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23.error
   |   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23.out
   |   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23.py
   |   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23.sh
   |   |   |-- forplots_california_housing_data_impact_n_plevel_34_43_23.submit
   |   |   |-- forplots_california_housing_data_impact_n_plevel_54_37_9.error
   |   |   |-- forplots_california_housing_data_impact_n_plevel_54_37_9.out
   |   |   |-- forplots_california_housing_data_impact_n_plevel_54_37_9.py
   |   |   |-- forplots_california_housing_data_impact_n_plevel_54_37_9.sh
   |   |   |-- forplots_california_housing_data_impact_n_plevel_54_37_9.submit
   |   |-- insurance
   |   |   |-- impact_eps_epsm
   |   |   |   |-- forplots_insurance_data_impact_epsc_epsm_plevel_34_43_23.py
   |   |   |-- impact_fc
   |   |   |   |-- forplots_insurance_data_impact_fc_plevel_34_43_23.py
   |   |   |-- impact_lambda
   |   |   |   |-- forplots_insurance_data_impact_lambda_plevel_34_43_23.py
   |   |   |-- impact_n
   |   |   |   |-- forplots_insurance_data_impact_n_plevel_34_43_23.py
   |   |   |   |-- forplots_insurance_data_impact_n_plevel_54_37_9.py
   |   |   |-- insurance_premium_34_43_23.py
   |   |   |-- insurance_premium_54_37_9.py
   |-- src
   |   |-- __pycache__
   |   |   |-- estimator.cpython-311.pyc
   |   |   |-- preprocessing.cpython-311.pyc
   |   |   |-- utils.cpython-311.pyc
   |   |-- estimator.py
   |   |-- preprocessing.py
   |   |-- utils.py
   |-- synthetic_expt
   |   |-- .DS_Store
   |   |-- csv_outputs
   |   |   |-- forplots_specific_30_100_eps_c_eps_m.csv
   |   |   |-- forplots_specific_30_100_eps_c_eps_m_CHECK.csv
   |   |   |-- forplots_specific_30_100_impact_fc.csv
   |   |   |-- forplots_specific_30_100_impact_n.csv
   |   |-- forplots_specific_d_n_plevel_34_43_23.py
   |   |-- forplots_specific_d_n_plevel_54_37_9.py
   |   |-- impact_epsc_epsm
   |   |   |-- CHECK.py
   |   |   |-- forplots_specific_d_n_epsc_epsm.py
   |   |-- impact_fc
   |   |   |-- forplots_specific_d_n_impact_fc.py
   |   |-- impact_n
   |   |   |-- forplots_specific_d_n_impact_n.py
   |   |-- run_plevel_34_43_23.py
   |   |-- run_plevel_54_37_9.py
   |-- tests
   |   |-- test_utils.py
```