from pymatgen.core import Structure
from pymatgen.io.cssr import Cssr as CssrPmg
import subprocess
from monty.os import cd,makedirs_p
import warnings
import re
import tempfile
import os
import warnings


class Cssr(CssrPmg):
    def __init__(self, structure):
        super().__init__(structure)

    def __str__(self):
        output = [
            "{:.4f} {:.4f} {:.4f}".format(*self.structure.lattice.abc),
            "{:.2f} {:.2f} {:.2f} SPGR =  1 P 1    OPT = 1".format(*self.structure.lattice.angles),
            f"{len(self.structure)} 0",
            f"0 {self.structure.formula}"
        ]
        for i, site in enumerate(self.structure.sites):
            output.append(f"{i + 1} {site.specie} {site.a:.4f} {site.b:.4f} {site.c:.4f} 0 0 0 0 0 0 0 0 0.00")
        return "\n".join(output)

class Zeopp:
    def __init__(self, structure: Structure, network_path: str = 'network'):
        """
        This class reproduce the behaviour of Zeo++ (http://www.zeoplusplus.org/).
        To get the various properties just use get_(property name).
        Check http://www.zeoplusplus.org/examples.html for documentation

        It stores the various properties in a dictionary that can be returned method by method.
        If the output file from zeo++ is needed, a path to store the file is needed. 
        The default is to remove the output file at the end.

        Args:
        
        structure: pymatgen Structure object that one wants to analyse.
        network_path: path to where zeo++ is installed, you must include ./network at the end of the path
        """
        self.network_path = network_path
        self.structure = structure
        
    def get_pore_diameter(self, out_dir: str = None, use_ha: bool = True):
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        
        input_struct = os.path.join(out_dir,'s.cssr') 
        output_file = os.path.join(out_dir,'out.res')
        Cssr(self.structure).write_file(filename=input_struct)
        
        command = [self.network_path,'-res', output_file, input_struct]
      
        if use_ha:
            command.insert(1,'-ha')
        process = subprocess.Popen(command)
        process.communicate()
        rrr = process.returncode
        if rrr != 0:
            raise RuntimeError('Process failed')
        output = {}
        prop = ["included_sphere","free_sphere","included_sphere_free_path"]
        
        
        with open(output_file,'r') as k:
            l = k.read()
            
            ll = re.compile(os.path.join(out_dir,'out.res    '))
            m = ll.split(l)
            val = m[1].split()
            
            for i,p in enumerate(prop):
                output[p] = val[i]
        
        return output 

    def get_channel_id_dim(self, out_dir: str = None, use_ha: bool = True, prob_radius: float = 1.5):
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        
        input_struct = os.path.join(out_dir,'s.cssr')
        output_file = os.path.join(out_dir,'out.chan')
        Cssr(self.structure).write_file(filename=input_struct)
 
        command = [self.network_path,'-chan', str(prob_radius), output_file, input_struct]

        if use_ha:
            command.insert(1,'-ha')
       
        process = subprocess.Popen(command)
        process.communicate()
        rrr = process.returncode
        if rrr != 0:
            raise RuntimeError('Process failed')

        prop = ["included_sphere_channel","free_sphere_channel","included_sphere_free_path_channel"]
        
        output = {}
        
        with open(output_file, 'r') as k:
            l = k.read()

            i = 0
            ll = re.compile("Channel  {0}  ".format(i))
            m = ll.split(l)
            
            for p in prop : 
                output[p] = []

            while len(m) == 2:
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                output["included_sphere_channel"].append(float(j.group()))

                ll = re.compile(f"{output['included_sphere_channel'][i]}  ")
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                output["free_sphere_channel"].append(float(j.group()))

                ll = re.compile(f"{output['free_sphere_channel'][i]}  ")
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                output["included_sphere_free_path_channel"].append(float(j.group()))

                i += 1
                ll = re.compile("Channel  {0}  ".format(i))
                m = ll.split(l)

        

        return output


    def get_surface_area(self, out_dir: str = None, use_ha: bool = True,probe_radius: float = 1.2, chan_radius: float = 1.2,
                         num_samples: int = 2000):
        if out_dir is None:
            out_dir = tempfile.mkdtemp()

        input_struct = os.path.join(out_dir,'s.cssr')
        output_file = os.path.join(out_dir,'out.sa')

        Cssr(self.structure).write_file(filename=input_struct)

        command = [self.network_path,'-sa', str(chan_radius), str(probe_radius), str(num_samples), output_file, 
                   input_struct]

        if use_ha:
            command.insert(1,'-ha')

        process = subprocess.Popen(command)
        process.communicate()
        rrr = process.returncode
        if rrr != 0:
            raise RuntimeError('Process failed')

        
        with open(output_file, 'r') as k:
            l = k.read()

            prop = ['Unitcell_volume: ', 'Density: ', 'ASA_A\^2: ', 'NASA_A\^2: ', 'Channel_surface_area_A\^2: ',
                    'Pocket_surface_area_A\^2: ']
            names = ['Unitcell_volume', 'Density', 'Acc_area', 'Not_acc_area','Channel_surf_area','Pocket_surface_area']

            
            output = {}
           
            for i, p in enumerate(prop): 
                n = 0
                if names[i] == 'Pocket_surface_area' or names[i] == 'Channel_surf_area':
                    if names[i] == 'Pocket_surface_area':
                        t = 'pockets'
                    else:
                        t = 'channels'
                    ll = re.compile('Number_of_{0}: '.format(t))
                    m = ll.split(l)
                    h = re.compile(r"(\d+)")
                    j = h.match(m[1])
                    n = int(j.group())
                    if n == 0:
                        output[names[i]] = 0.0
                        continue
                
                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                
                if n > 1:
                    output[names[i]] = []                                    
                    ll = re.compile('  ')
                    m = ll.split(m[1])
                    for I in range(n):
                       output[names[i]].append(float(m[I]))
                else:
                    output[names[i]] = float(j.group())

        
        return output


    def get_accessible_volume(self, out_dir: str = None, use_ha: bool = True, chan_radius: float = 1.2, probe_radius: float = 1.2,
                              num_samples: int = 50000):
        if out_dir is None:
            out_dir = tempfile.mkdtemp()

        input_struct = os.path.join(out_dir,'s.cssr')
        output_file = os.path.join(out_dir,'out.vol')

        Cssr(self.structure).write_file(filename=input_struct)

        command = [self.network_path,'-vol', str(chan_radius), str(probe_radius), str(num_samples), output_file,
                   input_struct]

        if use_ha:
            command.insert(1,'-ha')

        process = subprocess.Popen(command)
        process.communicate()
        rrr = process.returncode
        if rrr != 0:
            raise RuntimeError('Process failed')
 
        with open(output_file, 'r') as k:
            l = k.read()

            prop = ['Unitcell_volume: ', 'Density: ', 'AV_A\^3: ', 'NAV_A\^3: ', 'Channel_volume_A\^3: ',
                    'Pocket_volume_A\^3: ']
            names = ['Unitcell_volume', 'Density', 'Acc_volume', 'Not_acc_volume','Channel_volume','Pocket_volume']
            
            
            output = {}
            for i, p in enumerate(prop):
                n = 0
                if names[i] == 'Pocket_volume' or names[i] == 'Channel_volume':
                    if names[i] == 'Pocket_volume':
                        t = 'pockets'
                    else:
                        t = 'channels'
                    ll = re.compile('Number_of_{0}: '.format(t))
                    m = ll.split(l)
                    h = re.compile(r"(\d+)")
                    j = h.match(m[1])
                    n = int(j.group())
                    if n == 0:
                        output[names[i]] = 0.0
                        continue

                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])

                if n > 1:
                    output[names[i]] = []
                    ll = re.compile('  ')
                    m = ll.split(m[1])
                    for I in range(n):
                       output[names[i]].append(float(m[I]))
                else:
                    output[names[i]] = float(j.group())
        
        return output
 



    def get_probe_occupiable_volume(self, out_dir: str = None, use_ha: bool = True, chan_radius: float = 1.2, probe_radius: float = 1.2,
                                    num_samples: int = 50000):
       

        if out_dir is None:
            out_dir = tempfile.mkdtemp()

        input_struct = os.path.join(out_dir,'s.cssr')
        output_file = os.path.join(out_dir,'out.volpo')

        Cssr(self.structure).write_file(filename=input_struct)

        command = [self.network_path,'-volpo', str(chan_radius), str(probe_radius), str(num_samples), output_file,
                   input_struct]

        if use_ha:
            command.insert(1,'-ha')

        process = subprocess.Popen(command)
        process.communicate()
        rrr = process.returncode
        if rrr != 0:
            raise RuntimeError('Process failed')


        with open(output_file, 'r') as k:
            l = k.read()
            prop = ['Unitcell_volume: ', 'Density: ', 'POAV_A\^3: ', 'PONAV_A\^3: ']
            names = ['Unitcell_volume', 'Density', 'Probe_occupibale_volume', 'Not_acc_probe_occupibale_volume']
            
            
            output = {}
            for i, p in enumerate(prop):
                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                output[names[i]] = float(j.group())
        
            return output


