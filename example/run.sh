iqpath=SASDA25.out
pdbpath=SASDA25_fit1_model1.pdb
outpath=result
sastbx.python ../main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --target_pdb $pdbpath

