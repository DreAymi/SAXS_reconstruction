import voxel2pdb
import pdb2voxel
from scitbx.array_family import flex
import align
import os
import numpy as np
from sastbx.zernike_model import pdb2zernike
import zalign
import map2iq

def write2pdb(group,rmax,output_folder,iq_file=None,target_pdb=None):
	if iq_file is not None:
		tar_iq_curve=np.loadtxt(iq_file,usecols=(0))
		tar_iq_curve=tar_iq_curve.reshape(-1,1)
	num=len(group)
	os.system('mkdir %s/sub2'%output_folder)
	os.system('mkdir %s/sub3'%output_folder)
	for ii in range(num):
		voxel2pdb.write_pdb(group[ii],'%s/sub2/%d.pdb'%(output_folder,ii),rmax)
	fix='%s/sub2/0.pdb'%output_folder
	data=[]
	for ii in range(num):
		mov='%s/sub2/%d.pdb'%(output_folder,ii)
		align.run(fix,mov,'%s/sub3/%d.pdb'%(output_folder,ii))
		voxel=pdb2voxel.run(['pdbfile=%s/sub3/%d.pdb'%(output_folder,ii)])
		data.append(voxel)
	data=np.array(data)
	data=np.mean(data,axis=0)
	ccp4data=np.copy(data)
	data=np.greater(data,0.3).astype(int)

	if iq_file is not None:
		iq_curve,exp_data=map2iq.run_get_voxel_iq(data,iq_file,rmax)
		iq_curve=np.array(iq_curve)
		#iq_curve=iq_curve/iq_curve[0]
		newiq_curve=np.concatenate((tar_iq_curve,iq_curve.reshape((-1,1))),axis=1)
		np.savetxt('%s/final_saxs.txt'%output_folder,newiq_curve)

	voxel2pdb.write_pdb(data,'%s/out.pdb'%output_folder,rmax)

	'''
	if iq_file is not None:
		out_voxel=pdb2voxel.run(['pdbfile=%s/out.pdb'%output_folder])
		out_curve,exp_data=map2iq.run_get_voxel_iq(out_voxel,iq_file,rmax)
		out_curve=np.array(out_curve)
		out_curve=out_curve/out_curve[0]
		newout_curve=np.concatenate((tar_iq_curve,out_curve.reshape((-1,1))),axis=1)
		np.savetxt('%s/final_pdb_saxs.txt'%output_folder,newout_curve)
	'''

	ccp4data=flex.double(ccp4data)
	pdb2zernike.ccp4_map_type(ccp4data, 15, rmax/0.9,file_name='%s/out.ccp4'%output_folder)
	shiftrmax=rmax/0.9
	args=['fix=%s/out.ccp4'%output_folder,'typef=ccp4','mov=%s/out.pdb'%output_folder,'rmax=%f'%shiftrmax]
	zalign.run(args,output_folder)
	os.system('rm -r %s/sub2'%output_folder)
	os.system('rm -r %s/sub3'%output_folder)
	if target_pdb is not None:
		args=['fix=%s/out.ccp4'%output_folder,'typef=ccp4','mov=%s'%target_pdb,'rmax=%f'%shiftrmax]
		zalign.run(args,output_folder)
	if 'sample.pdb' in os.listdir(output_folder):
		args=['fix=%s/out.ccp4'%output_folder,'typef=ccp4','mov=%s/sample.pdb'%output_folder,'rmax=%f'%shiftrmax]
		zalign.run(args,output_folder)
	
def write_single_pdb(group,rmax,output_folder,target_pdb=None):
	ccp4data=np.copy(group.astype(float))
	voxel2pdb.write_pdb(group,'%s/out.pdb'%output_folder,rmax)
	ccp4data=flex.double(ccp4data)
	pdb2zernike.ccp4_map_type(ccp4data, 15, rmax/0.9,file_name='%s/out.ccp4'%output_folder)
	shiftrmax=rmax/0.9
	args=['fix=%s/out.ccp4'%output_folder,'typef=ccp4','mov=%s/out.pdb'%output_folder,'rmax=%f'%shiftrmax]
	zalign.run(args,output_folder)
	if target_pdb is not None:
		args=['fix=%s/out.ccp4'%output_folder,'typef=ccp4','mov=%s'%target_pdb,'rmax=%f'%shiftrmax]
		zalign.run(args,output_folder)


def cal_cc(voxel_group,rmax,output_folder,target_pdb,iq_file=None):
	os.system('mkdir %s/temp'%output_folder)
	os.system('mkdir %s/temp1'%output_folder)
	os.system('mkdir %s/temp2'%output_folder)

	if iq_file is not None:
		os.system('mkdir %s/saxs_fit_data'%output_folder)
		tar_iq_curve=np.loadtxt(iq_file,usecols=(0))
		tar_iq_curve=tar_iq_curve.reshape(-1,1)

	num=voxel_group.shape[0]
	cc_mat=np.zeros(shape=(num,20))
	cc_mat_aver=np.zeros(shape=(num))
	for ii in range(num):
		for jj in range(20):
			voxel2pdb.write_pdb(voxel_group[ii,jj],'%s/temp/%d_%d.pdb'%(output_folder,ii,jj),rmax)
			if iq_file is not None and jj==0:
				iq_curve,exp_data=map2iq.run_get_voxel_iq(voxel_group[ii,jj],iq_file,rmax)
				iq_curve=np.array(iq_curve)
				iq_curve=iq_curve/iq_curve[0]
				newiq_curve=np.concatenate((tar_iq_curve,iq_curve.reshape((-1,1))),axis=1)
				np.savetxt('%s/saxs_fit_data/%d_generation_voxelsaxs.txt'%(output_folder,ii),newiq_curve)
			cc=align.run(fix=target_pdb,mov='%s/temp/%d_%d.pdb'%(output_folder,ii,jj))
			cc_mat[ii,jj]=cc 
			print ii,jj,'%.3f'%cc
		fix='%s/temp/%d_0.pdb'%(output_folder,ii)

		if iq_file is not None:
			out_voxel=pdb2voxel.run(['pdbfile=%s'%fix])
			out_curve,exp_data=map2iq.run_get_voxel_iq(out_voxel,iq_file,rmax)
			out_curve=np.array(out_curve)
			out_curve=out_curve/out_curve[0]
			newout_curve=np.concatenate((tar_iq_curve,out_curve.reshape((-1,1))),axis=1)
			np.savetxt('%s/saxs_fit_data/%d_generation_pdbsaxs.txt'%(output_folder,ii),newout_curve)

		data=[]
		for jj in range(20):
			mov='%s/temp/%d_%d.pdb'%(output_folder,ii,jj)
			align.run(fix,mov,'%s/temp1/%d_%d.pdb'%(output_folder,ii,jj))
			voxel=pdb2voxel.run(['pdbfile=%s/temp1/%d_%d.pdb'%(output_folder,ii,jj)])
			data.append(voxel)
		data=np.array(data)
		data=np.mean(data,axis=0)
		data=np.greater(data,0.3).astype(int)

		voxel2pdb.write_pdb(data,'%s/temp2/%d.pdb'%(output_folder,ii),rmax)

		cc_aver=align.run(fix=target_pdb,mov='%s/temp2/%d.pdb'%(output_folder,ii))
		cc_mat_aver[ii]=cc_aver 
		print ii,'%.3f'%cc_aver
	
	np.savetxt('%s/cc_mat.txt'%output_folder,cc_mat,fmt='%.3f')
	np.savetxt('%s/cc_mat_aver.txt'%output_folder,cc_mat_aver,fmt='%.3f')
		
	os.system('rm -rf %s/temp'%output_folder)
	os.system('rm -rf %s/temp1'%output_folder)
	os.system('rm -rf %s/temp2'%output_folder)

	
