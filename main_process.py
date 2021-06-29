import numpy as np
import tensorflow as tf 
import exceptions
import map2iq
import auto_encoder_t
import time
import multiprocessing
import threading
import region_search
import os
import sys
import result2pdb
import voxel2pdb
import argparse
from functools import partial
from sastbx.zernike_model import pdb2zernike
from scitbx.array_family import flex

GPU_NUM=1
BATCH_SIZE=10
group_init_parameter=np.loadtxt('genegroup_init_parameter_2.txt',delimiter=' ')
np.set_printoptions(precision=10,threshold=np.NaN)

parser=argparse.ArgumentParser()
parser.add_argument('--iq_path',help='path of iq_file',type=str)
parser.add_argument('--rmax',help='radius of the protein',default=0,type=float)
parser.add_argument('--output_folder',help='path of output file',type=str)
parser.add_argument('--target_pdb',help='path of target pdb file',default='None',type=str)
parser.add_argument('--rmax_start',help='start range of rmax',default=10,type=float)
parser.add_argument('--rmax_end',help='end range of rmax',default=300,type=float)
args=parser.parse_args()

class MyThread(threading.Thread):
	def __init__(self,func,args=()):
		super(MyThread,self).__init__()
		self.func=func
		self.args=args

	def run(self):
		self.result=self.func(*self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None

class evolution:

	def __init__(self,output_folder,mode,rmax_start,rmax_end):
		self.mode=mode
		self.rmax_start=rmax_start
		self.rmax_end=rmax_end
		self.output_folder=output_folder
		self.iteration_step=0
		self.counter=0
		self.gene_length=200
		self.exchange_gene_num=2
		self.group_num=300
		self.inheritance_num=300
		self.remain_best_num=20
		self.statistics_num=20
		self.compute_score_withoutrmax=map2iq.run_withoutrmax
		self.compute_score_withrmax=map2iq.run_withrmax
		
		self.group=self.generate_original_group(self.group_num)
		self.group_score=self.compute_group_score(self.group)
		self.group,self.group_score=self.rank_group(self.group,self.group_score)

		if self.mode=='withoutrmax':
			self.topXnum=int(100)
			self.topXrmax=np.copy(self.group[:self.topXnum,-1])

		self.best_so_far=np.copy(self.group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.group_score[:self.remain_best_num])
		self.score_mat=np.copy(self.group_score[:self.statistics_num]).reshape((1,self.statistics_num))
		if self.mode=='withoutrmax':
			self.gene_data=np.copy(self.group[:self.statistics_num]).reshape((1,self.statistics_num,201))
		else:
			self.gene_data=np.copy(self.group[:self.statistics_num]).reshape((1,self.statistics_num,200))

		print 'original input , top5:',self.group_score[:5]
		print 'best_so_far, top5:',self.best_so_far_score[:5]
		print 'mean_score is:',np.mean(self.group_score)
		print 'initialized'

	def region_process(self,cube_group,indexs):
		num=cube_group.shape[0]
		z_group=[]
		real_data_group=[]
		for ii in range(num):
			out=cube_group[ii]
			while True:
				in_=np.zeros(shape=(1,32,32,32,1))
				in_[0,:31,:31,:31,0]=out
				z_,out_=sess.run([z_tensor_find[indexs],out_tensor_find[indexs]],feed_dict={in_tensor_find[indexs]:in_})
				z_=z_.reshape((self.gene_length))
				real_data=np.greater(out_[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(int)
				in_size=sum(real_data.reshape(-1))
				out,region_num=region_search.find_biggest_region(real_data)
				out_size=sum(out.reshape(-1))
				
				if region_num<=1:
					break
				
			z_group.append(z_.reshape(1,self.gene_length))
			real_data_group.append(real_data.reshape(1,31,31,31))
		z_group=np.concatenate(z_group,axis=0)
		real_data_group=np.concatenate(real_data_group,axis=0)
		return [z_group,real_data_group]

	def run_decode(self,data,ii):
		num=data.shape[0]//BATCH_SIZE
		rungroup=data.reshape((-1,BATCH_SIZE,self.gene_length))
		result=[]
		for jj in range(num):
			sub_result=sess.run(out_tensor[ii],feed_dict={in_tensor[ii]:rungroup[jj]})
			result.append(sub_result)
		result=np.concatenate(result,axis=0)
		result=np.greater(result,0.1).astype(int)
		result=result[:,:31,:31,:31,0].reshape((-1,31,31,31))
		return result

	def multi_thread_decode_group(self,group):
		source_num=group.shape[0]
		data_size=group.shape[0]//BATCH_SIZE
		if group.shape[0]%BATCH_SIZE!=0:
			data_size+=1
			proupcopy=np.copy(group)
			proupcopy.resize(((data_size)*BATCH_SIZE,self.gene_length))
			group=proupcopy
		real_data_group=[]
		sub_size=data_size//GPU_NUM
		addone_num=data_size%GPU_NUM
		threads=[]
		for kk in range(addone_num):
			t=MyThread(self.run_decode,args=(group[(kk*(sub_size+1)*BATCH_SIZE):((kk+1)*(sub_size+1)*BATCH_SIZE)],kk))
			threads.append(t)
			t.start()
		end_position=addone_num*(sub_size+1)*BATCH_SIZE
		if sub_size!=0:
			for ii in range(GPU_NUM-addone_num):
				t=MyThread(self.run_decode,args=(group[end_position+(ii*sub_size*BATCH_SIZE):end_position+((ii+1)*sub_size*BATCH_SIZE)],addone_num+ii))
				threads.append(t)
				t.start()
		for t in threads:
			t.join()
		for t in threads:
			sub_real_data_group=t.get_result()
			real_data_group.append(sub_real_data_group)
		real_data_group=np.concatenate(real_data_group,axis=0)
		return real_data_group[:source_num,:,:,:]

	def compute_group_score(self,group):
		decodetime1=time.time()
		real_data_group=self.multi_thread_decode_group(group[:,:self.gene_length])
		decodetime2=time.time()
		logfile.write('decode_time:%d\n'%(decodetime2-decodetime1))

		num=group.shape[0]
		if self.mode=='withoutrmax':
			group_rmax=np.copy(group[:,-1]).reshape(-1,1)
		
		t1=time.time()
		region_inf=np.empty(shape=(num),dtype=np.bool)
		pool=multiprocessing.Pool(processes=20)
		result=pool.map(region_search.find_biggest_region,real_data_group)
		pool.close()
		pool.join()
		

		for ii in range(num):
			if result[ii][1]>1:
				region_inf[ii]=True
				real_data_group[ii]=result[ii][0]
			else:
				region_inf[ii]=False
		
		process_gene_num=len([i for i in region_inf if i==True])
		logfile.write('reprocess gene number: %d\n'%process_gene_num)

		data_to_process=real_data_group[region_inf]
		data_to_process_z=group[region_inf]

		if self.mode=='withoutrmax':
			data_to_process_rmax=data_to_process_z[:,-1].reshape(-1,1)

		data_unchanged=real_data_group[(1 - region_inf).astype(bool)]
		z_unchanged=group[(1 - region_inf).astype(bool)]
		if data_to_process.shape[0]==0:
			real_data_group=data_unchanged
			self.group=z_unchanged
			
		else:
			threads=[]
			sub_group_num=data_to_process.shape[0]//GPU_NUM
			addone_num=data_to_process.shape[0]%GPU_NUM
			data_processed=[]
			z_processed=[]
			for ii in range(addone_num):
				t=MyThread(self.region_process,args=(data_to_process[(ii*(sub_group_num+1)):((ii+1)*(sub_group_num+1))],ii))
				threads.append(t)
				t.start()
			end_position=addone_num*(sub_group_num+1)
			if sub_group_num!=0:
				for jj in range(GPU_NUM-addone_num):
					t=MyThread(self.region_process,args=(data_to_process[(end_position+jj*sub_group_num):(end_position+(jj+1)*sub_group_num)],addone_num+jj))
					threads.append(t)
					t.start()
			for t in threads:
				t.join()
			for t in threads:
				result=t.get_result()
				data_processed.append(result[1])
				z_processed.append(result[0])
			data_processed=np.concatenate(data_processed,axis=0)
			z_processed=np.concatenate(z_processed,axis=0)

			if self.mode=='withoutrmax':
				z_processed=np.concatenate([z_processed,data_to_process_rmax],axis=1)

			real_data_group=np.concatenate([data_unchanged,data_processed],axis=0)
			self.group=np.concatenate([z_unchanged,z_processed],axis=0)

		
		self.save_all_voxelresult(real_data_group)
		

		t2=time.time()
		logfile.write('find_region_time:%d\n'%(t2-t1))
		compute_score_time1=time.time()
		
		if self.mode=='withoutrmax':
			compute_score_input_voxel=real_data_group.reshape(real_data_group.shape[0],-1)
			compute_score_input=np.concatenate([compute_score_input_voxel,self.group[:,-1].reshape(-1,1)],axis=1)
			pool=multiprocessing.Pool(processes=20)
			result=np.array(pool.map(self.compute_score_withoutrmax,compute_score_input))
			pool.close()
			pool.join()
			group_score=np.copy(result[:,0])
			self.group[:,-1]=result[:,1]
			
		elif self.mode=='withrmax':
			pool=multiprocessing.Pool(processes=20)
			result=pool.map(self.compute_score_withrmax,real_data_group)
			pool.close()
			pool.join()
			group_score=np.array(result)

		compute_score_time2=time.time()
		logfile.write('compute_score_time:%d\n'%(compute_score_time2-compute_score_time1))
	
		return group_score

	def rank_group(self,group,group_score):
		index=np.argsort(group_score)
		group=group[index]
		group_score=group_score[index]
		return group,group_score

	#two-point crossing
	def exchange_gene(self,selective_gene):
		np.random.shuffle(selective_gene)
		for ii in range(0,self.inheritance_num-self.remain_best_num,2):
			cross_point=np.random.randint(0,self.gene_length,size=(2*self.exchange_gene_num))
			cross_point=np.sort(cross_point)
			for jj in range(self.exchange_gene_num):
				random_data=np.random.uniform(low=0,high=1)
				if random_data<0.8:
					temp=np.copy(selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]])
					selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]]=selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]			
					selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]=np.copy(temp)

	#mutation oprator
	def gene_variation(self,selective_gene):
		if self.mode=='withoutrmax':
			average_rmax=np.mean(self.topXrmax)
			std_rmax=np.std(self.topXrmax)
		for ii in range(self.inheritance_num-self.remain_best_num):
			random_data=np.random.uniform(low=0,high=1,size=(self.gene_length+1))
			for jj in range(self.gene_length):
				if random_data[jj]<0.05:
					gene_point=np.random.normal(group_init_parameter[jj,0],group_init_parameter[jj,1],size=1)
					gene_point=abs(gene_point)
					selective_gene[ii,jj]=gene_point

			if self.mode=='withoutrmax':
				if random_data[-1]<0.5:
					random_num=np.random.uniform(low=0,high=1,size=1)
					if random_num<0.5:
						rmax_variation=np.random.randint(self.rmax_start,self.rmax_end)
						selective_gene[ii,-1]=rmax_variation
					else: 
						rmax_variation=np.random.normal(average_rmax,std_rmax,size=1)
						while rmax_variation<=10:
							rmax_variation=np.random.normal(average_rmax,std_rmax,size=1)
						selective_gene[ii,-1]=rmax_variation

	#select oprator
	def select_group(self):
		if self.mode=='withoutrmax':
			selected_group=np.zeros(shape=(self.inheritance_num-self.remain_best_num,self.gene_length+1))
		elif self.mode=='withrmax':
			selected_group=np.zeros(shape=(self.inheritance_num-self.remain_best_num,self.gene_length))

		selected_group_score=np.zeros(shape=(self.inheritance_num-self.remain_best_num))
		for ii in range(self.inheritance_num-self.remain_best_num):
			a=np.random.randint(0,self.group_num)
			b=np.random.randint(0,self.group_num)
			random_data=np.random.uniform(low=0,high=1)
			if random_data>0.1:
				if a<b:
					selected_group[ii]=np.copy(self.group[a])
					selected_group_score[ii]=np.copy(self.group_score[a])
				else:
					selected_group[ii]=np.copy(self.group[b])
					selected_group_score[ii]=np.copy(self.group_score[b])
			else:
				if a<b:
					selected_group[ii]=np.copy(self.group[b])
					selected_group_score[ii]=np.copy(self.group_score[b])
				else:
					selected_group[ii]=np.copy(self.group[a])
					selected_group_score[ii]=np.copy(self.group_score[a])
	
		self.group=selected_group
		self.group_score=selected_group_score

	def inheritance(self):
		self.select_group()
		self.exchange_gene(self.group)
		self.gene_variation(self.group)
		if self.group.shape[0]!=self.inheritance_num-self.remain_best_num:
			raise Exception('bad')
		self.group=np.concatenate((self.group,self.best_so_far),axis=0)
		t1=time.time()
		self.group_score=self.compute_group_score(self.group)
		t2=time.time()
		logfile.write('compute_group_score cost:%d\n'%(t2-t1))
		
		self.group,self.group_score=self.rank_group(self.group,self.group_score)
		if self.mode=='withoutrmax':
			self.topXrmax=np.copy(self.group[:self.topXnum,-1])
			self.gene_data=np.concatenate((self.gene_data,self.group[:self.statistics_num].reshape((1,self.statistics_num,201))),axis=0)
		elif self.mode=='withrmax':
			self.gene_data=np.concatenate((self.gene_data,self.group[:self.statistics_num].reshape((1,self.statistics_num,200))),axis=0)
		self.score_mat=np.concatenate((self.score_mat,self.group_score[:self.statistics_num].reshape((1,self.statistics_num))),axis=0)
		self.best_so_far=np.copy(self.group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.group_score[:self.remain_best_num])
		self.group=np.copy(self.group[:self.group_num])
		self.group_score=np.copy(self.group_score[:self.group_num])

	def save_latentspace(self):
		latendvalue=np.mean(self.group[:,:200],axis=0)
		np.save("%s/latentsteps_all/step%d.npy"%(self.output_folder,self.iteration_step),latendvalue)
		latend_ccp4_all=sess.run(sigleout_tensor[0],feed_dict={siglez_tensor[0]:latendvalue.reshape(1,200)})

		#latend_ccp4_all=np.greater(latend_ccp4_all[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(int)
		#print 'latend_ccp4_all',latend_ccp4_all.shape,type(latend_ccp4_all[0,0,0])
		#voxel2pdb.write_pdb(latend_ccp4_all,'%s/latentsteps_all/all_latentstep%d.pdb'%(self.output_folder,self.iteration_step),rmax)

		ccp4data_l_all=flex.double(np.greater(latend_ccp4_all[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(float))
		pdb2zernike.ccp4_map_type(ccp4data_l_all, 15, rmax/0.9,file_name='%s/latentsteps_all/all_latentstep%d.ccp4'%(self.output_folder,self.iteration_step))
		


		latendvaluetop20=np.mean(self.group[:20,:200],axis=0)
		np.save("%s/latentsteps_top20/step%d.npy"%(self.output_folder,self.iteration_step),latendvaluetop20)
		latend_ccp4_top20=sess.run(sigleout_tensor[0],feed_dict={siglez_tensor[0]:latendvaluetop20.reshape(1,200)})
		#latend_ccp4_top20=np.greater(latend_ccp4_top20[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(int)
		#voxel2pdb.write_pdb(latend_ccp4_top20,'%s/latentsteps_top20/top20_latentstep%d.pdb'%(self.output_folder,self.iteration_step),rmax)
		ccp4data_l_top20=flex.double(np.greater(latend_ccp4_top20[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(float))
		pdb2zernike.ccp4_map_type(ccp4data_l_top20, 15, rmax/0.9,file_name='%s/latentsteps_top20/top20_latentstep%d.ccp4'%(self.output_folder,self.iteration_step))

	def save_all_voxelresult(self,data):
		average_value=np.mean(data,axis=0)
		all_std=np.std(data,axis=0)
		np.save("%s/voxel_all/all_averagestep%d.npy"%(self.output_folder,self.iteration_step),average_value)
		np.save("%s/voxel_all/all_stdstep%d.npy"%(self.output_folder,self.iteration_step),all_std)
		ccp4data=flex.double(average_value)
		pdb2zernike.ccp4_map_type(ccp4data, 15, rmax/0.9,file_name='%s/voxel_all/all_averagestep%d.ccp4'%(self.output_folder,self.iteration_step))


	def evolution_iteration(self):
		while True:
			t1=time.time()
			self.save_latentspace()

			self.inheritance()
			self.iteration_step=self.iteration_step+1
			t2=time.time()
			print 'iteration_step:',self.iteration_step,'top5:',self.group_score[:5],'\nmean_score is:%.2f'%np.mean(self.score_mat[-1]),self.group_num
			logfile.write('iteration_step_%d'%self.iteration_step)
			logfile.write(' cost:%d \n\n'%(t2-t1))
			if self.score_mat[-1,0]<self.score_mat[-2,0]:
				self.counter=0
			else:
				self.counter=self.counter+1
				if self.counter>15:
					self.group_num=self.group_num-100
					self.inheritance_num=self.inheritance_num-100
					self.counter=0
					if self.group_num<100:
						#np.save('%s/score_mat.npy'%self.output_folder,self.score_mat)
						np.savetxt('%s/score_mat.txt'%self.output_folder,self.score_mat,fmt='%.3f')
						result_sample=self.multi_thread_decode_group(self.group[:self.statistics_num,:self.gene_length])
						t3=time.time()
						for kk in range(len(self.gene_data)):
							out_i=sess.run(sigleout_tensor[0],feed_dict={siglez_tensor[0]:self.gene_data[kk,0,:200].reshape(1,200)})
							np.save("%s/voxel_best/best_step%d.npy"%(self.output_folder,kk),out_i[0,:31,:31,:31,0].reshape((31,31,31)))
							#real_data_i=np.greater(out_i[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(int)
							#voxel2pdb.write_pdb(real_data_i,'%s/voxel_best/best_step%d.pdb'%(self.output_folder,kk),rmax)
							real_data_i=np.greater(out_i[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(float)
							ccp4data_best=flex.double(real_data_i)
							pdb2zernike.ccp4_map_type(ccp4data_best, 15, rmax/0.9,file_name='%s/voxel_best/best_step%d.ccp4'%(self.output_folder,kk))

						interpolation_value_start=np.mean(self.gene_data[0,:,:200].reshape(20,200),axis=0)
						interpolation_value_end=np.mean(self.gene_data[-1,:,:200].reshape(20,200),axis=0)
						for ll in range(101):
							interpolation_value=interpolation_value_start+(interpolation_value_end-interpolation_value_start)*(ll/100)
							inter_out=sess.run(sigleout_tensor[0],feed_dict={siglez_tensor[0]:interpolation_value.reshape(1,200)})
							ccp4_inter_out=np.greater(inter_out[0,:31,:31,:31,0].reshape((31,31,31)),0.1).astype(float)
							ccp4_inter_out=flex.double(ccp4_inter_out)
							pdb2zernike.ccp4_map_type(ccp4_inter_out, 15, rmax/0.9,file_name='%s/interpolation/step%d.ccp4'%(self.output_folder,ll))

						if self.mode=='withoutrmax':
							gene=self.gene_data.reshape((-1,self.gene_length+1))
							voxel_group=self.multi_thread_decode_group(gene[:,:-1])
							voxel_group=voxel_group.reshape((-1,self.statistics_num,31,31,31))	
							t4=time.time()
							logfile.write('\nvoxel_group cost:%d\n'%(t4-t3))
							np.savetxt('%s/bestgene.txt'%output_folder,self.group[0],fmt='%.3f')
							return result_sample,voxel_group,self.group[:self.statistics_num,-1],gene[:,-1].reshape((-1,self.statistics_num))
						else:
							gene=self.gene_data.reshape((-1,self.gene_length))
							voxel_group=self.multi_thread_decode_group(gene)
							voxel_group=voxel_group.reshape((-1,self.statistics_num,31,31,31))	
							t4=time.time()
							logfile.write('\nvoxel_group cost:%d\n'%(t4-t3))
							np.savetxt('%s/bestgene.txt'%output_folder,self.group[0],fmt='%.3f')
							return result_sample,voxel_group

	def generate_original_group(self,num):
		original_group=np.zeros(shape=(num,200))
		for ii in range(200):
			original_group[:,ii]=np.random.normal(group_init_parameter[ii,0],group_init_parameter[ii,1],size=num)
		original_group=abs(original_group)
		if self.mode=='withoutrmax':
			original_rmax=np.random.randint(self.rmax_start,self.rmax_end,(num,1)).astype(float)
			original_group=np.concatenate([original_group,original_rmax],axis=1)
		return original_group


if __name__=='__main__':
	
	iq_path=args.iq_path
	rmax=args.rmax
	real_rmax=args.rmax
	output_folder=args.output_folder
	target_pdb=args.target_pdb
	rmax_start=args.rmax_start
	rmax_end=args.rmax_end+1
	map2iq.iq_path=iq_path
	
	saved_model_path='model'
	auto_encoder_t.BATCH_SIZE=BATCH_SIZE
	map2iq.output_folder=output_folder
	
	logfile=open('%s/log.txt'%output_folder,'a')
	t1=time.time()
	generate_time1=time.time()
	in_tensor_find,z_tensor_find,out_tensor_find=auto_encoder_t.generate_session(GPU_NUM)
	in_tensor,out_tensor=auto_encoder_t.generate_session_decode(GPU_NUM)
	siglez_tensor,sigleout_tensor=auto_encoder_t.generate_decode_session(GPU_NUM)
	generate_time2=time.time()
	logfile.write('generate time:%d\n'%(generate_time2-generate_time1))
	print 'generate computing graph'

	evolution_mode=''
	if rmax==0:
		evolution_mode='withoutrmax'
	else:
		evolution_mode='withrmax'
	saver=tf.train.Saver()

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		model_path=tf.train.latest_checkpoint(saved_model_path)
		saver.restore(sess, model_path)
		genetic_object=evolution(output_folder,evolution_mode,rmax_start,rmax_end)
		if evolution_mode=='withoutrmax':
			result_sample,voxel_group,result_sample_rmax,voxel_group_rmax=genetic_object.evolution_iteration()
			t2=time.time()
			print 'result_sample_rmax:',result_sample_rmax
			rmax=np.mean(result_sample_rmax)
			print 'rmax_find',rmax
			np.savetxt('%s/rmax_find_log.txt'%output_folder,voxel_group_rmax,fmt='%d')
		else:
			map2iq.rmax=rmax
			print 'rmax_real:',rmax
			result_sample,voxel_group=genetic_object.evolution_iteration()
			
			average_top20_value=np.mean(voxel_group,axis=1)
			top20_std=np.std(voxel_group,axis=1)
			for ii in range(len(average_top20_value)):
				np.save("%s/voxel_top20/top20_averagestep%d.npy"%(output_folder,ii),average_top20_value[ii])
				np.save("%s/voxel_top20/top20_stdstep%d.npy"%(output_folder,ii),top20_std[ii])
				ccp4data_top20=flex.double(average_top20_value[ii])
				pdb2zernike.ccp4_map_type(ccp4data_top20, 15, rmax/0.9,file_name='%s/voxel_top20/top20_averagestep%d.ccp4'%(output_folder,ii))

			t2=time.time()

	if target_pdb == 'None':
		result2pdb.write2pdb( result_sample ,rmax ,output_folder,iq_path)
		t3=time.time()
	else:
		result2pdb.write2pdb( result_sample ,rmax ,output_folder,iq_path,target_pdb)
		t3=time.time()
		result2pdb.cal_cc(voxel_group,rmax,output_folder,target_pdb,iq_path)
	
	t4=time.time()
	print 'total_time:',(t4-t1)
	logfile.write('evolution time: %d\n'%(t2-t1))
	logfile.write('write2pdb time: %d\n'%(t3-t2))
	logfile.write('cal_cc time: %d\n'%(t4-t3))
	logfile.write('total time: %d\n'%(t4-t1))
	
	logfile.close()
	
