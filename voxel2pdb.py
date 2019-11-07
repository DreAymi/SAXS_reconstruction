import numpy as np
#np.set_printoptions(threshold=np.NaN)



def write_pdb(voxel,output,rmax):

	rmax=rmax/0.9
	f=open(output,'w')
	atom_id=0

	data=np.array([[1,1,1]])

	for x in range(31):
		for y in range(31):
			for z in range(31):
				if voxel[x,y,z]!=0:
					data=np.append(data,[[x,y,z]],axis=0)
	data=np.delete(data,0,axis=0)
	center=np.mean(data,axis=0)
	radius=np.max(np.sqrt(np.sum(np.square(data-center),axis=1)))/0.9
	num=data.shape[0]
	global r
	for atom_id in range(num):
		new_xyz=(data[atom_id]-center)/radius*rmax
		#new_xyz=data[atom_id]
		print>>f,"ATOM  %5d  C   ALA  %4d    %8.3f%8.3f%8.3f  1.00  1.00"%(atom_id+1, atom_id+1, new_xyz[0],new_xyz[1],new_xyz[2])

def voxel_to_pdb_as_string(voxel,rmax):

	rmax=rmax/0.9
	atom_id=0

	data=np.array([[1,1,1]])

	for x in range(31):
		for y in range(31):
			for z in range(31):
				if voxel[x,y,z]!=0:
					data=np.append(data,[[x,y,z]],axis=0)
	data=np.delete(data,0,axis=0)
	center=np.mean(data,axis=0)
	radius=np.max(np.sqrt(np.sum(np.square(data-center),axis=1)))/0.9
	num=data.shape[0]
	result=''
	global r
	for atom_id in range(num):
		new_xyz=(data[atom_id]-center)/radius*rmax
		#new_xyz=data[atom_id]
		result=result+"ATOM  %5d  C   ALA  %4d    %8.3f%8.3f%8.3f  1.00  1.00\n"%(atom_id+1, atom_id+1, new_xyz[0],new_xyz[1],new_xyz[2])
	return result	


