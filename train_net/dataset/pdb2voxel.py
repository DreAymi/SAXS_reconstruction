import sys, os
from stdlib import math as smath
from scitbx.array_family import flex
from libtbx.utils import Sorry, date_and_time, multi_out
import iotbx.phil
from iotbx import pdb, ccp4_map
from iotbx.option_parser import option_parser
import libtbx.phil.command_line
from cStringIO import StringIO
from libtbx.utils import null_out
from cctbx.eltbx import xray_scattering
from sastbx.data_reduction import saxs_read_write
import time
from mmtbx.monomer_library import server, pdb_interpretation

from scitbx import math
from sastbx import zernike_model as zm
from iotbx.xplor import map as xplor_map
from cctbx import uctbx, sgtbx

from libtbx import easy_pickle
from sastbx.interface import get_input

import numpy as np 

#f=file('voxel.npy','wb')

master_params = iotbx.phil.parse("""\
zernike{
  pdbfile = None
  .type=path
  .help="the pdb file"

  qmax = 0.3
  .type=float
  .help="maximum q value, for which the intensity to be evaluated"

  nmax=20
  .type=int
  .help="maximum order of zernike expansion"

  np=30
  .type=int
  .help="number of point covering [0,1]"

  fix_dx = False
  .type=bool
  .help="Whether the dx will be adjusted to match the size of molecule"

  buildmap=False
  .type=bool
  .help="Whether xplor map will be constructed or not"

  shift = False
  .type=bool
  .help="Whether the pdb coordates will be shifted or not"

  coef_out=True
  .type=bool
  .help="Whether dump zernike moments to picle files"
}
""")


banner = "--------------Zernike Moments Calculation and Map Construction----------------"

def help( out=None ):
  if out is None:
    out= sys.stdout
  print >> out, "Usage: libtbx.python pdb2zernike.py pdbfile=pdbfile nmax=nmax buildmap=True/False shift=True/False np=np_of_point"


def write_bead( original_map , N):
    threshold = 0.5
#    this_grid = flex.grid(N*2+1, N*2+1, N*2+1)
    my_select = flex.bool( original_map.as_1d() >= threshold )

    cube=np.zeros(shape=(2*N+1 , 2*N+1 , 2*N+1))
    indx = 0
    atom_id = 0
    for x in range(-N,N+1):
      for y in range(-N,N+1):
        for z in range(-N,N+1):
          if my_select[ indx ] :
            atom_id = atom_id + 1
            cube[x+N][y+N][z+N]=1
          indx = indx + 1
#    np.save(f,cube)
    return cube


def ccp4_map_type(map, N, radius,file_name='map.ccp4'):
  grid = flex.grid(N*2+1, N*2+1,N*2+1)
  print N,'\n'
  map.reshape( grid )
  ccp4_map.write_ccp4_map(
      file_name=file_name,
      unit_cell=uctbx.unit_cell(" %s"%(radius*2.0)*3+"90 90 90"),
      space_group=sgtbx.space_group_info("P1").group(),
      gridding_first=(0,0,0),
      gridding_last=(N*2, N*2, N*2),
      map_data=map,
      labels=flex.std_string(["generated from zernike moments"]))



def xplor_map_type(m,N,radius,file_name='map.xplor'):
  gridding = xplor_map.gridding( [N*2+1]*3, [0]*3, [2*N]*3)
  grid = flex.grid(N*2+1, N*2+1,N*2+1)
  m.reshape( grid )
  uc = uctbx.unit_cell(" %s"%(radius*2.0)*3+"90 90 90")
  xplor_map.writer( file_name, ['no title lines'],uc, gridding,m) # is_p1_cell=True)  # True)


def zernike_moments(pdbfile, nmax=20, fix_dx=False, np_on_grid=15, shift=False, buildmap=False, coef_out=True, calc_intensity=True, external_rmax=-1):
  base = pdbfile.split('.')[0]
  splat_range = 0
  fraction = 0.9
  default_dx = 0.7
  uniform = True

  pdbi = pdb.hierarchy.input(file_name=pdbfile)
  if(len( pdbi.hierarchy.models() ) == 0):
    return None,None,None

  atoms = pdbi.hierarchy.models()[0].atoms()
  # predefine some arrays we will need
  atom_types = flex.std_string()
  radius= flex.double()
  b_values = flex.double()
  occs = flex.double()
  xyz = flex.vec3_double()
  # keep track of the atom types we have encountered
  for atom in atoms:
    if(not atom.hetero):
      xyz.append( atom.xyz )
#    b_values.append( atom.b )
#    occs.append( atom.occ )

  if(xyz.size() == 0):
    return None,None,None
  density=flex.double(xyz.size(),1.0)
  voxel_obj = math.sphere_voxel(np_on_grid,splat_range,uniform,fix_dx,external_rmax, default_dx, fraction,xyz,density)
  np = voxel_obj.np()
  print 'np',np
  rmax=voxel_obj.rmax()/fraction
  print 'rmax',rmax

  #print base, "RMAX: ", voxel_obj.rmax()
  original_map = voxel_obj.map()
  rmax = np
  #ccp4_map_type( original_map, np, rmax, file_name=base+'_pdb.ccp4')
  cube=write_bead( original_map, np)
  return cube

####### The following will be optional ###

  if(shift):
    shift = [rmax, rmax, rmax]
    centered_xyz = voxel_obj.xyz() + shift
    out_pdb_name=base+'_centered.pdb'
    for a,xyz in zip( atoms, centered_xyz):
      a.set_xyz( new_xyz=xyz)
    pdbi.hierarchy.write_pdb_file( file_name=out_pdb_name, open_append=False)




def run(args):
  params = get_input(args, master_params, "zernike", banner, help)
  if params is None:
    return
  pdbfile = params.zernike.pdbfile
  nmax=params.zernike.nmax
  np = params.zernike.np
  fix_dx = params.zernike.fix_dx
  shift = params.zernike.shift
  buildmap = params.zernike.buildmap

  cube=zernike_moments(pdbfile, nmax, fix_dx=fix_dx, shift=shift)
  return cube


if __name__ == "__main__":
  '''
  pdbname_file=open('args.txt','r')
  pdbname=pdbname_file.readlines()
  for ii in range(len(pdbname)):
    run([pdbname[ii]])
  '''
  args=sys.argv[1:]
  cube=run(args)
  np.save('voxel.npy',cube)

