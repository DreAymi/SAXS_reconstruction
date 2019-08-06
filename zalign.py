from libtbx import easy_pickle
from stdlib import math as smath
from scitbx.array_family import flex
import os, sys
from sastbx.zernike_model import model_interface
from scitbx import math, matrix
from iotbx import pdb

import iotbx.phil
from scitbx.math import zernike_align_fft as fft_align
from iotbx.xplor import map as xplor_map
from cctbx import uctbx
from scitbx.golden_section_search import gss
import time

from sastbx.interface import get_input


master_params = iotbx.phil.parse("""\
align{
  fix = None
  .type=path
  .help="pickle Cnlm coef of the fixed object"

  mov = None
  .type=path
  .help="pickle Cnlm coef of the fixed object"

  typef=*pdb nlm xplor ccp4 mrc
  .type=choice
  .help="fixed model type, default PDB"

  typem=*pdb nlm xplor ccp4 mrc 
  .type=choice
  .help="moving model type, default PDB"

  num_grid = 41
  .type=int
  .help="number of point in each euler angle dimension"

  rmax = None
  .type=float
  .help="maxium radial distance to the C.o.M. (before the scaling)"

  nmax = 20
  .type=int
  .help="maximum order of zernike polynomial:fixed for the existing database"

  topn = 10
  .type=int
  .help="top N alignments will be further refined if required"

  refine = True
  .type=bool
  .help="Refine the initial alignments or not"

  write_map = False
  .type=bool
  .help="write xplor map to file"


}
""")

banner = "-------------------Align molecular models in various format (PDB/NLM/XPLOR) ---------------------"

def get_mean_sigma( nlm_array ):
  coef = nlm_array.coefs()
  mean = abs( coef[0] )
  var = flex.sum( flex.norm(coef) )
  sigma = smath.sqrt( var-mean*mean )
  return mean, sigma


def run(args,outpath=None):
  params = get_input(args, master_params, "align", banner, help)
  if params is None:
    return
  fix = params.align.fix
  typef = params.align.typef
  mov = params.align.mov
  typem = params.align.typem

  num_grid = params.align.num_grid
  nmax = params.align.nmax
  rmax = params.align.rmax
  topn = params.align.topn
  write_map = params.align.write_map

  #outpath=params.align.outpath

  fix_model=model_interface.build_model( fix, typef, nmax, rmax )
  mov_model=model_interface.build_model( mov, typem, nmax, rmax )

  #fix_nl_array = fix_model.nl_array
  #mov_nl_array = mov_model.nl_array

  #CHI2 = flex.sum_sq( fix_nl_array.coefs() - mov_nl_array.coefs() )
  
  #print "CHI2 between Fnl's is %e\n"%CHI2

  fix_nlm_array = fix_model.nlm_array
  mov_nlm_array = mov_model.nlm_array
  print "doing alignment"
  align_obj = fft_align.align( fix_nlm_array, mov_nlm_array, nmax=nmax, topn=topn ,refine=True)

  cc = align_obj.get_cc()

  mov_model.nlm_array = align_obj.moving_nlm

  rmax = update_rmax( rmax, fix_model, mov_model)
  fix_model.rmax = rmax
  mov_model.rmax = rmax

  shift=(rmax, rmax, rmax)
  
  print "#############  SUMMARY of ALIGNMENT  #############"
  print "Correlation Coefficient Between two models is: ", cc
  print "Rmax is                                      : ", rmax
  print "Euler angles for the moving object is        : ", list(align_obj.best_ea)
  print "Center of Mass is shifted to                 : ", list(shift)
  print "OUTPUT files are : "
  
  current_is_mov = False
  for model in (fix_model, mov_model):
    #base = model.id
    base=outpath+'/'+mov.split("/")[-1].split(".")[0]
    '''
    easy_pickle.dump(base+"_za.nlm", model.nlm_array.coefs() )
    print "  "+base+"_za.nlm"
    '''
    if(current_is_mov): model.map=None
    if(write_map):
      model.write_map(filename=base+"_za.xplor")
      print "   "+base+"_za.xplor"

    if( model.vox_obj is not None):  ### Write aligned PDB file ####
      out_pdb_name=base+"_za.pdb"
      if(current_is_mov):
        ea = align_obj.best_ea
        aligned_xyz = model.vox_obj.rotate((-ea[0],ea[1],-ea[2]), False)
      else:
        aligned_xyz = model.vox_obj.xyz()

      aligned_xyz = aligned_xyz + shift  ### Add the shift, such that the EDM center is the same as PDB
      model.pdb_inp.hierarchy.atoms().set_xyz(aligned_xyz)
      model.pdb_inp.hierarchy.write_pdb_file( file_name=out_pdb_name, open_append=False)
      print "  "+out_pdb_name
    current_is_mov = True
  print "#############     END of SUMMARY     #############"


def update_rmax( rmax, fix_model, mov_model):
  if rmax is None:
    if( fix_model.pdb_inp is not None):
      rmax = fix_model.rmax
      print "RMAX was None, and is set to be the same as pdb structure"
    elif( mov_model.pdb_inp is not None):
      rmax = mov_model.rmax
      print "RMAX was None, and is set to be the same as pdb structure"
    elif( rmax is None):
      rmax = 50
      print "RMAX was None, and is set to be the same, default value 50A"
  return rmax

def help( out=None ):
  if out is None:
    out= sys.stdout
  print >> out, "\nUsage: \nsastbx.superpose fix=fixed_file typef=type [pdb | nlm | map ] mov=moving_file typem=type nmax=nmax\n"

if __name__=="__main__":
  args = sys.argv[1:]
  t1 = time.time()
  run(args)
  t2 = time.time()
  print "total time used: ", t2-t1
