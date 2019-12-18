iqpath=SASDA25.out
pdbpath=SASDA25_fit1_model1.pdb
outpath=result
model_path=../model/
sastbx.python ../main.py > $outpath/logprint.txt --model_path $model_path --iq_path $iqpath --output_folder $outpath --target_pdb $pdbpath

