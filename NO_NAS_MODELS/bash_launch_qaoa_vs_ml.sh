for exe in {1..30}; 
do 
	mkdir -p Models/qaoa_vs_ml/execution_${exe}
	  cp -r Models/ml_vs_ml/MLP/Execution_${exe}/Models/*.pkl Models/qaoa_vs_ml/execution_${exe}
	  cp -r Models/ml_vs_ml/xgboost/Execution_${exe}/Models/*.pkl Models/qaoa_vs_ml/execution_${exe}
	  python Classical_ML_Comp.py ${exe}
done
