# RandomCarbon

## Introduction

This repository contains the RandomCarbon Python code. This package is used to randomly generate symmetric carbon structures around a template. The package was developed to generate schwarzite models as described in [Marazzi et. al, "Modelling symmetric and defect-free carbon schwarzites into various zeolite templates"](URL). Please refer to the pubblication for more details.

## Aim

The code is intended to generate new models for schwarzites [1-3]. Schwarzites are 3D carbon-based nanostructures which are characterized by a negative curvature and that form 3 bonds with other carbon atoms (sp<sup>2</sup> hybridization). These nanostructures are 'surface-like' hence, once the reference template has been chosen, they are predicted to occupy a narrow space in a certain distance from it. Therefore, one constraint is that the carbon atoms distance from the template <i>d</i> will be <i>min_distance &lt; d &lt; max_distance</i>. Another constraint is the hybridization, so models in which carbon atoms form exactly three bonds will be promoted. Finally, the goal is to produce symmetrical structures: as a consequence, a spacegroup will be enforced and kept during the all generation process. To deal with structure generation and the spacegroup analysis the [pymatgen](https://pymatgen.org/) package is used. 

The templates provided in the [templates](randomcarbon/data/templates/) folder are some of zeolite from the [IZA Database of Zeolite structure](http://www.iza-structure.org/databases/), however one can also use other templates from that database or other sources.


## Installation

In order to get the RandomCarbon package, clone the repository:

```
git clone https://github.com/modl-uclouvain/randomcarbon.git
```
For pip use:
```
pip install -r requirements.txt
```

If you're using [conda](https://conda.io/en/latest/) create a new environment (`randomcarbon`) based on python3.9:

```
conda create -n randomcarbon python=3.9
source activate randomcarbon
```

and install the dependencies:

```
conda install --file ./requirements.txt
```

Once the requirements have been installed (either with pip or conda), execute:
```
python setup.py install
```

or, for developing:
```
python setup.py develop
```

## Random generation

The random generation consists in adding atoms, one by one, to the unit cell. While doing this the symmetry spacegroup need to be conserved. The evolution of the structures can be performed in different ways through the [evolvers](randomcarbon/evolution/evolvers/). This random generation can be driven in the good direction imposing some constraint on the generated structures with the help of [blockers](randomcarbon/evolution/blockers), [conditions](randomcarbon/evolution/conditions) and [filters](randomcarbon/evolution/filters). The main constraints are the distance from the template, the number of nearest neighbors and the maximum energy per atom.

The generated structures are saved in Json format as python dictionaries and may be stored either locally or in a MongoDB database through [pymongo](https://pymongo.readthedocs.io/en/stable/).

## Genetic algorithm

The genetic algorithm is based on the one implemented in [ASE](https://wiki.fysik.dtu.dk/ase/ase/ga/ga.html). The growth, reduction and mixing of the structures operations have been adapted to fit with the requirements of the present work. With respect to the operations available for the random generation, here some evolvers that move atoms, and merge two or more structures have been added.

## Templates geometry

Two tools have been employed to analyze the template geometry:

 - The [tubes part](randomcarbon/tubes/search.py) can be used to look for the largest channel radius in different directions for a given input structure, or for the largest channel radius of the whole system.

 - The [zeopp part](randomcarbon/zeopp/zeopp.py) allows to execute and parse the outputs of the Open Source code [Zeo++](http://www.zeoplusplus.org/).

The aim of these tools is to analyze the used template and, on the one hand, find a relationship between the energy of the generated structure and the templates features, one the other, to compare the newly founded models with already-synthesized nanostructures.


### Acknowledgement

The authors acknowledge fundings from the European Union’s Horizon 2020 Research Project and Innovation Program — Graphene Flagship Core3 (N° 881603), from the Fédération Wallonie-Bruxelles through the ARC on Dynamically Reconfigurable Moiré Materials (N° 21/26-116), and from the Belgium F.R.S.-FNRS through the research project (N° T.029.22F). Computational resources have been provided by the CISM supercomputing facilities of UCLouvain and the CÉCI consortium funded by F.R.S.-FNRS of Belgium (N° 2.5020.11)


### Authors

Enrico Marazzi, Ali Ghojavand, Jérémie Pirard, Guido Petretto, Jean-Christophe Charlier, Gian-Marco Rignanese

### References

[1] L Mackay, H Terrones, Diamond from graphite. Nature 352, 762 (1991)

[2] D Vanderbilt, J Tersoff, Negative-curvature fullerene analog of c60. Phys. Rev. Lett. 68, 511–513 (1992)

[3] T Lenosky, X Gonze, M Teter, V Elser, Energetics of negatively curved graphitic carbon. Nature 355, 333–335 (1992)

### License

RandomCarbon is released under the MIT License. For more details see the [LICENSE](LICENSE) file.
