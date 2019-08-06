iqpath=SASDAH6.out
pdbpath=SASDAH6_fit1_model1.pdb
outpath=result
sastbx.python ../main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --rmax 55.5 --target_pdb $pdbpath
#CUDA_VISIBLE_DEVICES=1 sastbx.python ../main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --rmax 55.5 --target_pdb $pdbpath
