#!/bin/bash
#PBS -l procs=4
#PBS -l walltime=8:00:00
#PBS -l mem=60GB
#PBS -N fit-kic
#PBS -M danfm@nyu.edu
#PBS -j oe

module purge
export PATH="$HOME/miniconda3/bin:$PATH"
module load mvapich2/intel/2.0rc1
export OMP_NUM_THREADS=1

export ISOCHRONES=$SCRATCH/isochrones_data

SRCDIR=$HOME/projects/gaia-kepler
export PATH="$SRCDIR:$PATH"

RUNDIR=$SCRATCH/gaia-kepler/results
mkdir -p $RUNDIR

cd $RUNDIR
mpiexec -np $PBS_NP python $SRCDIR/fit.py

