# Prioritizing with importance weighting in Semi-Supervised Variational Autoencoders
This repository serves reproducing the experiments from the paper 
"Controlling the Interaction Between Generation and Inference in 
Semi-Supervised Variational Autoencoders Using Importance Weighting".

The commands for running the set of experiment for each model and each dataset
is in the corresponding ```.slurm``` or ```.sh``` file
 named ```launch<Model><Dataset>```.
  
 For importance weighted experiments, the memory cost 
 can be lowered using the ```--grad_accu``` parameter which specifies the number 
 of gradient accumulation steps. The ```--batch_size``` parameter must then be lowered 
 accordingly. Example: ```--grad_accu 2 --batch_size 32``` is equivalent to 
 ```--grad_accu 4 --batch_size 16``` .