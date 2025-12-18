#BSUB -J data_CM_CfD_EoM_Storage
#BSUB -oo data_CM_CfD_EoM_Storage.out
#BSUB -eo data_CM_CfD_EoM_Storage.err
#BSUB -q p_short
#BSUB -n 1
#BSUB -R span[ptile=1]
#BSUB -M 10GB
#BSUB -P 0588

module purge

source /dir/miniconda3/initialize_miniconda.sh

conda activate ray_3

python data_CM_CfD_EoM_Storage.py           

