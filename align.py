from libtbx import easy_pickle
from stdlib import math as smath
from scitbx.array_family import flex
import os, sys
from sastbx.zernike_model import model_interface
from scitbx import math, matrix
from iotbx import pdb
from sastbx.zernike_model import build_pymol_script

import iotbx.phil
from scitbx.math import zernike_align_fft as fft_align
from iotbx.xplor import map as xplor_map
from cctbx import uctbx
from scitbx.golden_section_search import gss
import time

from sastbx.interface import get_input

base_path = os.path.split(sys.path[0])[0]

global targetfile
master_params = iotbx.phil.parse("""\
align{
  fix = None
  .type=path
  .help="pickle Cnlm coef of the fixed object"

  mov = None
  .type=path
  .help="pickle Cnlm coef of the fixed object"

  typef=*pdb nlm map
  .type=choice
  .help="fixed model type, default PDB"

  typem=*pdb nlm map
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
output = "output"
.type=path
.help = "Output base. expect a .pr and a .qii file."
""")

banner = "-------------------Align molecular models in various format (PDB/NLM/XPLOR) ---------------------"

def get_mean_sigma( nlm_array ):
  coef = nlm_array.coefs()
  mean = abs( coef[0] )
  var = flex.sum( flex.norm(coef) )
  sigma = smath.sqrt( var-mean*mean )
  return mean, sigma

# def start(args):
#   last_arg = args[-1]
#   targetfile = str(str(last_arg).split("=")[-1])
#   args.pop()
#   run(args,targetfile)

def run(fix,mov,out_pdb_name=None):
  #targetfile = $SASTBXPATH/modules/cctbx_project/sastbx
  '''
  targetfile = os.path.join(os.path.split(sys.path[0])[0],"superpose.txt")
  with open(targetfile,"w") as f:
    f.truncate()
  
  tempf = open(targetfile,'w')
  params = get_input(args, master_params, "align", banner, help,tempf)
  tempf.close()
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

  print fix,typef,mov,typem,num_grid,nmax,rmax,topn,write_map
  '''

  fix_model=model_interface.build_model( fix, 'pdb', 20, None )
  mov_model=model_interface.build_model( mov, 'pdb', 20, None )


  fix_nlm_array = fix_model.nlm_array
  mov_nlm_array = mov_model.nlm_array
  '''
  with open(targetfile,"a") as f:
    f.write("doing alignment\n")
  print "doing alignment"
  '''
  align_obj = fft_align.align( fix_nlm_array, mov_nlm_array, nmax=20, topn=10 ,refine=True)

  cc = align_obj.get_cc()
  
  if out_pdb_name is not None:
    ea = align_obj.best_ea
    aligned_xyz = mov_model.vox_obj.rotate((-ea[0],ea[1],-ea[2]), False)
    #rmax = update_rmax( 1, fix_model, mov_model)
    rmax = fix_model.rmax
    shift=(rmax, rmax, rmax)
    aligned_xyz = aligned_xyz #+ shift  ### Add the shift, such that the EDM center is the same as PDB
    mov_model.pdb_inp.hierarchy.atoms().set_xyz(aligned_xyz)
    #out_pdb_name='aligned_'+mov
    mov_model.pdb_inp.hierarchy.write_pdb_file( file_name=out_pdb_name, open_append=False)
  
  # print cc
  return cc
  '''
  mov_model.nlm_array = align_obj.moving_nlm

  rmax = update_rmax( rmax, fix_model, mov_model)
  fix_model.rmax = rmax
  mov_model.rmax = rmax

  shift=(rmax, rmax, rmax)

  with open(targetfile,"a") as f:
    f.write( "#############  SUMMARY of ALIGNMENT  #############\n")
    f.write( "Correlation Coefficient Between two models is: "+str(cc)+"\n")
    f.write("Rmax is                                      : "+str(rmax)+"\n")
   
    f.write("Center of Mass is shifted to                 : "+str(list(shift))+"\n")   
    f.write("OUTPUT files are : "+"\n")
    

  
  print "#############  SUMMARY of ALIGNMENT  #############"
  print "Correlation Coefficient Between two models is: ", cc
  print "Rmax is                                      : ", rmax
  print "Center of Mass is shifted to                 : ", list(shift)
  print "OUTPUT files are : "

  current_is_mov = False

  pdblist = []
  xplorlist = []

  targetpath_fromGUI = ''
  targetpath_fromGUI_file =  os.path.join(base_path,"targetpath_GUI.txt")
  if os.path.isfile(targetpath_fromGUI_file) and (os.stat(targetpath_fromGUI_file).st_size>0):
    with open(targetpath_fromGUI_file,"r") as f:
      targetpath_fromGUI = f.read().strip()

  for model in (fix_model, mov_model):
    if targetpath_fromGUI == '':
      base=model.id
    else:
      
      base = str(model.id.split("/")[-1])
      print "base:   ",base
      targetdir = os.path.join(targetpath_fromGUI,"Model_Superposition")
      base = os.path.join(targetdir,base)


    ##################20170520###########################
    #################change the output dir ###################
    # base = str(model.id.split("/")[-1])
    # dirlist = sys.argv[0].split("sastbx")
    # tmpdir = str(dirlist[0])+"sastbx/gui/sasqt/tmp.txt"
    # with open(tmpdir,"r") as f:
    #   targetdir = str(f.read().strip())
    # base = os.path.join(targetdir,"superpose",base)
    ###############################################################
    easy_pickle.dump(base+"_za.nlm", model.nlm_array.coefs() )
    with open(targetfile,"a") as f:
      f.write("  "+base+"_za.nlm\n")
    
    if(write_map):
      model.write_map(filename=base+"_za.xplor")
      xplorlist.append(base+"_za.xplor")

      with open(targetfile,"a") as f:
        f.write("   "+base+"_za.xplor\n")
      

    if( model.vox_obj is not None):  ### Write aligned PDB file ####
      out_pdb_name=base+"_za.pdb"
      pdblist.append(out_pdb_name)


      if(current_is_mov):
        ea = align_obj.best_ea
        aligned_xyz = model.vox_obj.rotate((-ea[0],ea[1],-ea[2]), False)
      else:
        aligned_xyz = model.vox_obj.xyz()

      aligned_xyz = aligned_xyz + shift  ### Add the shift, such that the EDM center is the same as PDB
      ###################20170511#####################################
      ################debug for size error############################
      #model.pdb_inp.hierarchy.atoms().set_xyz(aligned_xyz)
      sel_cache = model.pdb_inp.hierarchy.atom_selection_cache()
      hetero = model.pdb_inp.hierarchy.atoms().extract_hetero()
      position = list(hetero)
      no_hetero = sel_cache.selection("all")
      for i in position:
          no_hetero[i]=False
      no_hetero_atoms = model.pdb_inp.hierarchy.atoms().select(no_hetero)
      no_hetero_atoms.set_xyz(aligned_xyz)
      
      model.pdb_inp.hierarchy.write_pdb_file( file_name=out_pdb_name, open_append=False)
      with open(targetfile,"a") as f:
        f.write("  "+out_pdb_name+'\n')

      print out_pdb_name
    
    current_is_mov = True
  # print "pdblist: ",pdblist
  # print "xplorlist: ", xplorlist

  ############targetpath_fromGUI=='' for commmand line 
  ############else for GUI

  if targetpath_fromGUI != '':
    targetdir = os.path.join(targetpath_fromGUI,"Model_Superposition")
    build_pymol_script.write_pymol_superpose(pdblist,targetdir)
    
  with open(targetfile,"a") as f:
    f.write("#############     END of SUMMARY     #############\n")

  with open(targetfile,"a") as f:
    f.write("__END__")

  print "#############     END of SUMMARY     #############\n"
  print "__END__"
  '''


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
  #start(args)
  run(*args)
  t2 = time.time()
  print "total time used: ", t2-t1
