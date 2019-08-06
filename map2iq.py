import sys, os, random
from stdlib import math as smath
from scitbx.array_family import flex
import iotbx.phil
from iotbx import pdb
from sastbx.data_reduction import saxs_read_write
from sastbx import zernike_model as zm
from sastbx.zernike_model.search_pdb import reduce_raw_data, get_mean_sigma
from sastbx.zernike_model import  model_interface
from scitbx.math import zernike_align_fft as fft_align
import time
from scitbx import math
from iotbx.xplor import map as xplor_map
from cctbx import uctbx
from libtbx import easy_pickle
from sastbx.interface import get_input
import voxel2pdb
import numpy as np
import copy

iq_path=''
output_folder=''
rmax=50

master_params = iotbx.phil.parse("""\
zrefine{
  target = None
  .type=path
  .help="the experimental intensity profile"

  start = None
  .multiple=True
  .type=path
  .help="starting model in xplor format or pickle file of coefs"

  pdb = None
  .type=path
  .help="PDB model to be compared"

  rmax = None
  .type=float
  .help="estimated rmax of the molecule"

  qmax = 0.15
  .type=float
  .help="maximum q value where data beyond are disgarded"

  n_trial=5
  .type=int
  .help="number of refinement trials"

  nmax=10
  .type=int
  .help="maximum order of zernike expansion"

  splat_range=1
  .type=int
  .help="depth of alterable region from original surface"

  prefix='prefix'
  .type=path
  .help="prefix of the output files"
}
""")

banner = "-------------------Shape Refinement-------------------"

def xplor_map_type(m,N,radius,file_name='map.xplor'):
  gridding = xplor_map.gridding( [N*2+1]*3, [0]*3, [2*N]*3)
  grid = flex.grid(N*2+1, N*2+1,N*2+1)
  m.reshape( grid )
  uc = uctbx.unit_cell(" %s"%(radius*2.0)*3+"90 90 90")
  xplor_map.writer( file_name, ['no title lines'],uc, gridding,m) # is_p1_cell=True)  # True)

def help( out=None ):
  if out is None:
    out= sys.stdout
  print >> out, "Usage: libtbx.python zm_xplor_refine.py target=target.iq start=start.xplor rmax=rmax"

def construct(target_file,rmax):
  data = saxs_read_write.read_standard_ascii_qis( target_file )
  ed_map_obj = ED_map(data, rmax)
  return ed_map_obj

def normalize(data):
  #copydata=np.copy(data)
  data=np.array(data)
  dmax=data.max()
  data=data/dmax
  return data

def linear_fit(x,y,s): # Standard least square fitting
  var = s*s
  sum_x2 = flex.sum( x*x/var )
  #sum_y2 = flex.sum( y*y/var )
  sum_xy = flex.sum( x*y/var )
  sum_x  = flex.sum( x / var )
  sum_y  = flex.sum( y / var )
  N = x.size()
  sum_inv_var = flex.sum(1.0/var)
  det = sum_inv_var * sum_x2 - sum_x * sum_x
  scale = (sum_inv_var * sum_xy - sum_x*sum_y ) / det
  offset = (sum_x2*sum_y - sum_x * sum_xy) /det
  return scale, offset

def run_get_voxel_iq(start_file, iq_file=None, r=None):
  global ip_path
  global rmax
  if iq_file is None:
    iq_file=iq_path
  if r is None:
    r=rmax
  data = saxs_read_write.read_standard_ascii_qis( iq_file )
  sourcedata=copy.copy(data)
  ed_map_obj = ED_map(data, r)

  this_map=start_file.reshape((-1))
  calc_i = ed_map_obj.compute_saxs_profile(this_map)
  scale, offset = linear_fit( calc_i, data.i, data.s )
  calc_i = scale*calc_i+offset
  #calc_i = calc_i/calc_i[0]
  return calc_i,sourcedata

def run_withrmax(start_file, iq_file=None, rmax_center=None):
  
  global iq_path
  global rmax
  log_distance=[]
  if iq_file is None:
    iq_file=iq_path
  if rmax_center is None:
    rmax_center=rmax
  data = saxs_read_write.read_standard_ascii_qis( iq_file )
  this_map=start_file.reshape((-1))

  ed_map_obj = ED_map(data, rmax_center)
  distance=ed_map_obj.target(this_map)
  return distance

def run_withoutrmax(start_file, iq_file=None, rmax_center=None):
  global iq_path
  log_distance=[]
  start_file=start_file.reshape((-1))
  if iq_file is None:
    iq_file=iq_path
  if rmax_center is None:
    rmax_center=start_file[-1]
  data = saxs_read_write.read_standard_ascii_qis( iq_file )
  this_map=start_file[:-1]
  
  if rmax_center>10:
    choosermax=rmax_center
    bestdistance=200
    for ii in range(int(rmax_center-6),int(rmax_center+7),3):
      ed_map_obj = ED_map(data, ii)
      distance=ed_map_obj.target(this_map)
      if distance<bestdistance:
        bestdistance=distance
        choosermax=ii
    return bestdistance,choosermax
  else:
    ed_map_obj = ED_map(data, rmax_center)
    distance=ed_map_obj.target(this_map)
    return distance,rmax_center


class ED_map(object):
  def __init__(self, data, rmax, qmax=0.15, nmax=20, np_on_grid=15, prefix='prefix', fraction=0.9):
    self.raw_data=data
    self.rmax = rmax/fraction
    self.fraction = fraction
    self.qmax = qmax
    self.nmax = nmax
    self.np_on_grid=np_on_grid
    self.ngp = (self.np_on_grid*2+1)**3
    self.all_index = flex.int(range(self.ngp))
    self.n = self.np_on_grid*2+1
    self.n2 = self.n**2
    self.prefix=prefix+'_'

    self.nlm_array=math.nlm_array(nmax)
    self.nlm=self.nlm_array.nlm()

    self.bandwidth = min( smath.pi/rmax/2.0, 0.01 )
    self.data = data
    self.scale_2_expt = self.data.i[0]
    self.initialize_reusable_objects()


  def load_maps(self, files): # take about 0.5s to load one map, not big deal
    xplor_file = files[0]
    this_xplor = xplor_map.reader(xplor_file)
    self.raw_map= this_xplor.data.as_1d()
    
    self.np_on_grid = int( (this_xplor.gridding.n[0]-1 ) /2 )


  def initialize_reusable_objects(self):
    #### Reusable Objects for moments calculation ####
    self.grid_obj=math.sphere_grid(self.np_on_grid, self.nmax)
    self.moment_obj = math.zernike_moments( self.grid_obj, self.nmax )
    moments = self.moment_obj.moments()
    self.zm = zm.zernike_model( moments, self.data.q, self.rmax, self.nmax)

  def build_sphere_list(self):
    self.indx_list=flex.int()
    indx_range = range(self.np_on_grid*2+1)
    np = self.np_on_grid
    np2 = np**2
    for ix in indx_range:
      for iy in indx_range:
        for iz in indx_range:
          if( (ix-np)**2 + (iy-np)**2 + (iz-np)**2 < np2 ):
            self.indx_list.append( self.convert_indx_3_1( (ix,iy,iz) ) )
    return


  def compute_saxs_profile(self, map):
    this_map = flex.double( map )
    std_deviation = this_map.standard_deviation_of_the_sample()
    threshold = flex.mean( this_map ) + std_deviation
    this_list = self.all_index.select(this_map > threshold)

    space_sum = self.grid_obj.construct_space_sum_via_list(this_list)
    self.moment_obj.calc_moments( space_sum.as_1d() )
    nn_obj = self.moment_obj.fnn()
    self.calc_i = self.zm.calc_intensity(nn_obj)
    return self.calc_i
 
  def target(self, map):
    map=map.reshape((-1))
    self.calc_i = self.compute_saxs_profile( map )
    self.calc_i=self.calc_i/self.calc_i[0]
    self.data.i=self.data.i/self.data.i[0]

    #scale, offset = linear_fit( self.calc_i, self.data.i, self.data.s )
    #self.calc_i = scale*self.calc_i+offset

    '''
    datasize_cal=len(self.calc_i)-1
    derivative_cal=np.zeros(shape=datasize_cal)
    for ii in range(datasize_cal):
      derivative_cal[ii]=self.calc_i[ii+1]-self.calc_i[ii]
    #derivative_cal=derivative_cal/derivative_cal[0]
    
    derivative_tar=np.zeros(shape=datasize_cal)
    for ii in range(datasize_cal):
      derivative_tar[ii]=self.data.i[ii+1]-self.data.i[ii]
    #derivative_tar=derivative_tar/derivative_tar[0]
    distance=(derivative_cal-derivative_tar)/derivative_tar

    weight=np.zeros(shape=len(distance),dtype=float)
    baseweight=np.array([1.0,0.8,0.7,0.6,0.5])
    subsize=len(distance)//5
    addone=len(distance)%5
    for ii in range(addone):
      weight[ii*(subsize+1):(ii+1)*(subsize+1)]=baseweight[ii]
    endpoint=addone*(subsize+1)
    for ii in range(5-addone):
      weight[endpoint+ii*subsize:endpoint+(ii+1)*subsize]=baseweight[addone+ii]
    
    #weight=np.linspace(1,0,len(distance))
    #weight[-1]=weight[-2]
    score = np.linalg.norm(distance*weight)
    #score = np.linalg.norm((derivative_cal-derivative_tar)/derivative_tar)
    '''
    #score = ((self.calc_i-self.data.i)/self.data.i).norm()
    score = (self.calc_i-self.data.i).norm()
    
    if np.isnan(score):
      return 200
    return score
  
if __name__ == "__main__":
  start_file=np.load('test.npy')
  start_file=np.copy(start_file[:31,:31,:31])
  print run(start_file)
    
   





