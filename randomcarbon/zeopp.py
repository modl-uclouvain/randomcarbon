from pymatgen.core import Structure
from pymatgen.io.cssr import Cssr
import subprocess
from monty.os import cd,makedirs_p
import warnings
import re


class Cssr1(Cssr):
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
    def __init__(self, network_path: str, structure: str):
        """
        This class reproduce the behaviour of Zeo++ (http://www.zeoplusplus.org/).
        To get the various properties just use get_(property name).
        Check http://www.zeoplusplus.org/examples.html for documentation

        It stores the various properties in a dictionary that is returned by every method

        Args:

        network_path: path to where zeo++ is installed, you must include ./network at the end of the path
        structure: string to the input structure one wants to analyse. The default format is .cssr, but it can handle
                   all the format that pymatgen Structure can handle
        """
        self.np = network_path     
        makedirs_p('tmp')
        self.dict = {}
        if structure.endswith('.cssr'):
            self.structure = structure
        else:
            s = Structure.from_file(structure)
            self.structure = 'tmp/s.cssr'
            Cssr1(s).write_file(filename=self.structure)

    def get_pore_diameter(self,use_ha: bool = True):
        tmp_output = 'tmp/out.res'
        if use_ha:
            command = [self.np,'-ha','-res',tmp_output, self.structure]
        else:
            command = [self.np,'-res', tmp_output, self.structure]
        subprocess.run(command)
        with open(tmp_output,'r') as k:
            l = k.read()
            
            ll = re.compile('tmp/out.res    ')
            m = ll.split(l)
            h = re.compile(r"(\d+)(\.(\d+))*")
            j = h.match(m[1])
            self.dict["included_sphere"] = float(j.group())
            
            ll = re.compile(' ')
            m1 = ll.split(m[1])
            self.dict["free_sphere"] = float(m1[1])
        
            ll = re.compile('  ')
            m2 = ll.split(m1[1])
            self.dict["included_sphere_free_path"] = float(m1[3])

        return self.dict  

    def get_channel_id_dim(self,use_ha: bool = True,prob_radius: float = 1.5):
        tmp_output = 'tmp/out.chan'
        if use_ha:
            command = [self.np, '-ha', '-chan',str(prob_radius), tmp_output, self.structure]
        else:
            command = [self.np, '-chan',str(prob_radius), tmp_output, self.structure]
        subprocess.run(command)
        with open(tmp_output, 'r') as k:
            l = k.read()

            i = 0
            ll = re.compile("Channel  {0}  ".format(i))
            m = ll.split(l)
            self.dict["included_sphere_channel"] = []
            self.dict["free_sphere_channel"] = []
            self.dict["included_sphere_free_path_channel"] = []
            while len(m) == 2:
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                self.dict["included_sphere_channel"].append(float(j.group()))

                ll = re.compile(f"{self.dict['included_sphere_channel'][i]}  ")
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                self.dict["free_sphere_channel"].append(float(j.group()))

                ll = re.compile(f"{self.dict['free_sphere_channel'][i]}  ")
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                self.dict["included_sphere_free_path_channel"].append(float(j.group()))

                i += 1
                ll = re.compile("Channel  {0}  ".format(i))
                m = ll.split(l)
        return self.dict


    def get_surface_area(self,use_ha: bool = True,probe_radius: float = 1.2, chan_radius: float = 1.2,
                         num_samples: int = 2000):
        tmp_output = 'tmp/out.sa'
        if use_ha:
            command = [self.np, '-ha', '-sa', str(chan_radius),str(probe_radius), str(num_samples), tmp_output,
                       self.structure]
        else:
            command = [self.np, '-sa', str(chan_radius),str(probe_radius), str(num_samples), tmp_output, self.structure]
        subprocess.run(command)
        with open(tmp_output, 'r') as k:
            l = k.read()

            prop = ['Unitcell_volume: ', 'Density: ', 'ASA_A\^2: ', 'NASA_A\^2: ', 'Channel_surface_area_A\^2: ',
                    'Pocket_surface_area_A\^2: ']
            names = ['Unitcell_volume', 'Density', 'Acc_area', 'Not_acc_area','Channel_surf_area','Pocket_surface_area']

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
                        self.dict[names[i]] = 0.0
                        continue
                
                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                
                if n > 1:
                    self.dict[names[i]] = []                                    
                    ll = re.compile('  ')
                    m = ll.split(m[1])
                    for I in range(n):
                       self.dict[names[i]].append(float(m[I]))
                else:
                    self.dict[names[i]] = float(j.group())
        return self.dict

    def get_accessible_volume(self, use_ha: bool = True, chan_radius: float = 1.2, probe_radius: float = 1.2,
                              num_samples: int = 50000):
        tmp_output = 'tmp/out.vol'
        if use_ha:
            command = [self.np, '-ha', '-vol', str(chan_radius),str(probe_radius), str(num_samples), tmp_output,
                       self.structure]
        else:
            command = [self.np, '-vol', str(chan_radius),str(probe_radius), str(num_samples), tmp_output,
                       self.structure]

        subprocess.run(command)
        with open(tmp_output, 'r') as k:
            l = k.read()

            prop = ['Unitcell_volume: ', 'Density: ', 'AV_A\^3: ', 'NAV_A\^3: ', 'Channel_volume_A\^3: ',
                    'Pocket_volume_A\^3: ']
            names = ['Unitcell_volume', 'Density', 'Acc_volume', 'Not_acc_volume','Channel_volume','Pocket_volume']
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
                        self.dict[names[i]] = 0.0
                        continue

                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])

                if n > 1:
                    self.dict[names[i]] = []
                    ll = re.compile('  ')
                    m = ll.split(m[1])
                    for I in range(n):
                       self.dict[names[i]].append(float(m[I]))
                else:
                    self.dict[names[i]] = float(j.group())
        return self.dict



    def get_probe_occupiable_volume(self, use_ha: bool = True, chan_radius: float = 1.2, probe_radius: float = 1.2,
                                    num_samples: int = 50000):
        tmp_output = 'tmp/out.volpo'
        if use_ha:
            command = [self.np, '-ha', '-volpo', str(chan_radius), str(probe_radius), str(num_samples), tmp_output,
                       self.structure]
        else:
            command = [self.np, '-vol', str(chan_radius), str(probe_radius), str(num_samples), tmp_output,
                       self.structure]

        subprocess.run(command)
        with open(tmp_output, 'r') as k:
            l = k.read()
            prop = ['Unitcell_volume: ', 'Density: ', 'POAV_A\^3: ', 'PONAV_A\^3: ']
            names = ['Unitcell_volume', 'Density', 'Probe_occupibale_volume', 'Not_acc_probe_occupibale_volume']

            for i, p in enumerate(prop):
                ll = re.compile(p)
                m = ll.split(l)
                h = re.compile(r"(\d+)(\.(\d+))*")
                j = h.match(m[1])
                self.dict[names[i]] = float(j.group())

        return self.dict
