# RandomCarbon

## Introduction

This repository contains the RandomCarbon Python code. This package is used to randomly generate symmetric carbon structures around a template. The generated structures will be required to satisfy certain constraints. The templates provided in [randomcarbon/data/templates/](randomcarbon/data/templates/) are zeolite structures taken from [IZA Database for Zeolite Sructures](https://www.iza-structures.org/databases/), but one can use other structures from the same database or some completely different. The package was developed to generate schwarzite models as described in [Marazzi et. al, "Modelling symmetric and defect-free carbon schwarzites into various zeolite templates"](URL). Please refere to the pubblication for more details.

## Installation

In order to get the RandomCarbon package, clone the repository:

```
git clone https://github.com/modl-uclouvain/randomcarbon.git
```
For pip use:
```
pip install -r requirements.txt
```

If you're using [conda](https://conda.io/en/latest/) create a new environment ('randomcarbon') based on python3.9:

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

The random generation consists in adding atoms, one by one, to the unit cell. While doing this the symmetry spacegroup need to be conserved. The evolution of the structures can be performed in different ways through the [evolvers](randomcarbon/evolution/evolvers/). This random generation can be driven in the good direction imposing some constraint on the generated structures with the help of [blockers](randomcarbon/evolution/blockers), [conditions](randomcarbon/evolution/conditions) and [filters](randomcarbon/evolution/filters). The main constraints are the distance from the template, the number of nearest neighbor and the maximum energy per atom.

The generated structures are saved in Json format as python dictionaries and may be stored either locally or online on mongodb through [pymongo](https://pymongo.readthedocs.io/en/stable/).

## Genetic algorithm

The genetic algorithm is based on the implementation of the genetic algorithm in [ASE](https://wiki.fysik.dtu.dk/ase/ase/ga/ga.html). The growth, reduction and mixing of the structures operations have been adapted to fit with the requirements of the present work. 

## Templates geometry

Two tools have been employed to analyze the template geometry:

 - The [tubes part](randomcarbon/tubes/search.py) can be used to look for the largest channel radius in different directions for a given input structure, or for the largest channel radius of the whole system.

 - The [zeopp part](randomcarbon/zeopp/zeopp.py) reproduces through a Python interface the behaviour of the Open Source code [Zeo++](http://www.zeoplusplus.org/).

The aim of these tools is to analyze the used template and, on the one hand, find a relationship between the energy of the generated structure and the templates features, one the other, to compare the newly founded models with already-synthesized nanostructures.


### Acknowledgement

The authors acknowledge fundings from the European Union’s Horizon 2020 Research Project and Innovation Program — Graphene Flagship Core3 (N° 881603), from the Fédération Wallonie-Bruxelles through the ARC on Dynamically Reconfigurable Moiré Materials (N° 21/26-116), and from the Belgium F.R.S.-FNRS through the research project (N° T.029.22F). Computational resources have been provided by the CISM supercomputing facilities of UCLouvain and the CÉCI consortium funded by F.R.S.-FNRS of Belgium (N° 2.5020.11)


### Authors

Enrico Marazzi, Ali Ghojavand, Jérémie Pirard, Guido Petretto, Jean-Christophe Charlier, Gian-Marco Rignanese

### License

RandomCarbon is released under the GNU GENERAL PUBLIC LICENSE. For more details see the LICENSE file.
