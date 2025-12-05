version_to_use=1
version_qaoa_vs_ml=1
for exe in {1..30};
do 
	  mkdir -p Models/qaoa_vs_ml/V${version_qaoa_vs_ml}/execution_${exe}
	  cp -r Models/ml_vs_ml/V${version_to_use}/execution_${exe}/Models/ml_vs_ml/mlp/*.pkl Models/qaoa_vs_ml/V${version_qaoa_vs_ml}/execution_${exe}
	  cp -r Models/ml_vs_ml/V${version_to_use}/execution_${exe}/Models/ml_vs_ml/xgboost/*.pkl Models/qaoa_vs_ml/V${version_qaoa_vs_ml}/execution_${exe}
	  #python Classical_ML_Comp_indiv.py ${exe}
    python Classical_ML_Comp_Full.py ${exe}
done
