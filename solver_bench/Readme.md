The benchmarking script was run for RTVD, Riemann split and Riemann unsplit
solvers and the results from the base size tests were gathered here in
respective files. To present the results run:

    ./bench_plot.py RTVD Riemann_split Riemann_unsplit

* Note that this is the performance per reported timestep, so the unsplit
  solver may be much slower in realistic applications because:
    * It requires lowering the CFL to 0.3 while the split solvers can work with CFL = 0.7.
    * It requires doubled amount of timestep because it was written in a single-step manner.
      The split solvers count double-timesteps (XYZ sweeps then ZYX sweeps).
* In the implicit diffusion applications one must reduce the value of
  `diff_tstep_fac` to account for single-step implementation to maintain
  accuracy similar to split solver applications.
