iqpath=SASDA25.out

outpath=result
sastbx.python main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath

#give rmax
#rmax=35
#sastbx.python main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --rmax $rmax

#if you want to compare with exist pdb structure
#pdbpath=SASDA25_fit1_model1.pdb
#sastbx.python main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --rmax 35 --target_pdb $pdbpath

#CUDA_VISIBLE_DEVICES=1 sastbx.python main.py > $outpath/logprint.txt --iq_path $iqpath --output_folder $outpath --rmax 55.5 --target_pdb $pdbpath
