# Optimization 

Numerical optimization is an indispensible part of machine learning. I'd in fact argue that it is the ONLY non-trivial aspect of Machine Learning (inviting the ire  of computer scientists here...). The following chart nicely describes the different kinds of unconstrained optimization algorithms we learned in this class: 

![An overview of some unconstrained optimization methods](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Screenshot%202021-01-20%20at%2011.36.11%20PM.png)

The implementations can be found below: 

1. First derivative algorithms
	* [Gradient Descent](https://github.com/ShashankSule/AMSC808N/blob/master/Final/Solutions%20to%20Final%20Part%201--The%20Descent%20of%20Gradient.ipynb)--A Julia Implementation can be found in Problem 3. 
	* [Stochastic Gradient Descent](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Shashank_Code/SGD.m)
	* [Conjugate Gradient](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Project1/CG.m)
 2. Second Derivative algorithms 
 	1. Line Search 
 		* [Gauss-Newton](https://github.com/ShashankSule/AMSC808N/blob/master/Problem%20Sets/PSET3/GaussNewton.m)
		* [Line search routine](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Shashank_Code/ls.m)
		* [Subsampled Inexact Newton](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Project1/SINewton.m)
	2. Trust Region 
		* [Levenberg-Marquadt](https://github.com/ShashankSule/AMSC808N/blob/master/Problem%20Sets/PSET3/LevenbergMarquardt.m)
		* [LBFGS](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Shashank_Code/lbfgs.m)
		* [SLBFGS](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Shashank_Code/SLBFGS.m)
		
As for constrained optimization, here is an implementation of the [Active Set Method](https://github.com/ShashankSule/AMSC808N/blob/master/Optimization/Project%201/Project1/ASM.m). 
