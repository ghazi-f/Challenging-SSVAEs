# Challenging the Semi-Supervised VAE Framework for Text Classification
This repository serves reproducing the experiments from the paper 
[Challenging the Semi-Supervised VAE Framework for Text 
Classification(link to be added)](https://arxiv.org/abs/2010.06549).

The commands for running the set of experiment for each model and each dataset
is in the corresponding ```.slurm``` or ```.sh``` file
 named ```launch<Model><Dataset>``` in the ```launch_scripts``` folder.
  
 For importance weighted experiments, the memory cost 
 can be lowered using the ```--grad_accu``` parameter which specifies the number 
 of gradient accumulation steps. The ```--batch_size``` parameter must then be lowered 
 accordingly. Example: ```--grad_accu 2 --batch_size 32``` is equivalent to 
 ```--grad_accu 4 --batch_size 16``` .