# scipy.optimize.differential_evolution¶

```
scipy.optimize.differential_evolution(func,bounds,args=(), strategy='best1bin', maxiter=1000,popsize=15, tol=0.01, mutation=(0.5, 1),recombination=0.7, seed=None, callback=None,disp=False, polish=True, init='latinhypercube',atol=0, updating='immediate', workers=1)[source]
```

Finds the global minimum of a multivariate function.

Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimium, and can search large areas of candidate space, but often requires larger numbers of function evaluations than conventional gradient based techniques.

The algorithm is due to Storn and Price [1].

## Parameters

* func : callable
  
        The objective function to be minimized. Must be in the form f(x, *args), where x is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function.

* bounds : sequence
  
        Bounds for variables. (min, max) pairs for each element in x, defining the lower and upper bounds for the optimizing argument of func. It is required to have len(bounds) == len(x). len(bounds) is used to determine the number of parameters in x.

* strategy : str, optional

        The differential evolution strategy to use. Should be one of:
        ‘best1bin’
        ‘best1exp’
        ‘rand1exp’
        ‘randtobest1exp’
        ‘currenttobest1exp’
        ‘best2exp’
        ‘rand2exp’
        ‘randtobest1bin’
        ‘currenttobest1bin’
        ‘best2bin’
        ‘rand2bin’
        ‘rand1bin’
        The default is ‘best1bin’.

* maxiter : int, optional

        The maximum number of generations over which the entire population is evolved. The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * len(x)

* popsize : int, optional

        A multiplier for setting the total population size. The population has popsize * len(x) individuals (unless the initial population is supplied via the init keyword).

* tol : float, optional

        Relative tolerance for convergence, the solving stops when np.std(pop) <= atol + tol * np.abs(np.mean(population_energies)), where and atol and tol are the absolute and relative tolerance respectively.

* mutation : float or tuple(float, float),optional

        The mutation constant. In the literature this is also known as differential weight, being denoted by F. If specified as a float it should be in the range [0, 2]. If specified as a tuple (min, max) dithering is employed. Dithering randomly changes the mutation constant on a generation by generation basis. The mutation constant for that generation is taken from U[min, max). Dithering can help speed convergence significantly. Increasing the mutation constant increases the search radius, but will slow down convergence.

* recombination : float, optional

        The recombination constant, should be in the range [0, 1]. In the literature this is also known as the crossover probability, being denoted by CR. Increasing this value allows a larger number of mutants to progress into the next generation, but at the risk of population stability.

* seed : int or np.random.RandomState, optional

        If seed is not specified the np.RandomState singleton is used. If seed is an int, a new np.random.RandomState instance is used, seeded with seed. If seed is already a np.random.RandomState instance, then that np.random.RandomState instance is used. Specify seed for repeatable minimizations.

        disp : bool, optional
        Display status messages

* callback : callable, callback(xk, convergence=val), optional

        A function to follow the progress of the minimization. xk is the current value of x0. val represents the fractional value of the population convergence. When val is greater than one the function halts. If callback returns True, then the minimization is halted (any polishing is still carried out).

* polish : bool, optional

        If True (default), then scipy.optimize.minimize with the L-BFGS-B method is used to polish the best population member at the end, which can improve the minimization slightly.

* init : str or array-like, optional

        Specify which type of population initialization is performed. Should be one of:

        ‘latinhypercube’
        ‘random’
        array specifying the initial population. The array should have shape (M, len(x)), where len(x) is the number of parameters. init is clipped to bounds before use.
        The default is ‘latinhypercube’. Latin Hypercube sampling tries to maximize coverage of the available parameter space. ‘random’ initializes the population randomly - this has the drawback that clustering can occur, preventing the whole of parameter space being covered. Use of an array to specify a population subset could be used, for example, to create a tight bunch of initial guesses in an location where the solution is known to exist, thereby reducing time for convergence.

* atol : float, optional

        Absolute tolerance for convergence, the solving stops when np.std(pop) <= atol + tol * np.abs(np.mean(population_energies)), where and atol and tol are the absolute and relative tolerance respectively.

* updating : {‘immediate’, ‘deferred’}, optional

        If 'immediate', the best solution vector is continuously updated within a single generation [4]. This can lead to faster convergence as trial vectors can take advantage of continuous improvements in the best solution. With 'deferred', the best solution vector is updated once per generation. Only 'deferred' is compatible with parallelization, and the workers keyword can over-ride this option.

        New in version 1.2.0.

* workers : int or map-like callable, optional

        If workers is an int the population is subdivided into workers sections and evaluated in parallel (uses multiprocessing.Pool). Supply -1 to use all available CPU cores. Alternatively supply a map-like callable, such as multiprocessing.Pool.map for evaluating the population in parallel. This evaluation is carried out as workers(func, iterable). This option will override the updating keyword to updating='deferred' if workers != 1. Requires that func be pickleable.

        New in version 1.2.0.

## Returns:	
* res : OptimizeResult
        The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination. See OptimizeResult for a description of other attributes. If polish was employed, and a lower minimum was obtained by the polishing, then OptimizeResult also contains the jac attribute.