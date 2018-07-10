

# Ion-trap Tomography Stimulation
tomo.py: general API for tomography

choose one from two:\
(1).Traditional methods
* curve_fit_class.py: class of traditional curve fitting methods for estimation of Rabi oscillation curve parameters 

(2).Neural Networks methods
* data_pick.py: generate data from theory for training the neural network predictor
* NNtrainer.py: trianing to get weights & bias for NNpredictor
* NNpredictor.py: the class of neural network predictor


choose one from two:\
(1).[Maximum Likelihood methods](https://arxiv.org/abs/1605.05039)
* measure.py: class of tomography functions

(2).[Iteration methods](https://arxiv.org/abs/quant-ph/0009093)
* ite.py: using  instead of in measure.py for tomography


## original neural network predictor for Rabi oscillation curve parameters:
full2.py: full2 network structure\
full2+BN.py: full2 network structure with Batch Normalization
