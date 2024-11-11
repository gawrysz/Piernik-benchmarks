Early Fedora 40 wass unusable in Piernik.

* openmpi-5.0.2-2.fc40 works only for some configurations.  It cannot grind
  the standard benchmarking set, crashing a lot on strong scaling.  For
  example in the sedov test it goes into MPI deadlock in initpiernik,
  because internal_boundaries_MPI_merged can't properly communicate data
  between 3 threads due to something messed up earlier, most likely in dot.

* mpich-4.1.2-14.fc40 doesn't work, even single thread doesn't run because

    Abort(272194565) on node 0 (rank 0 in comm 0): Fatal error in internal_Comm_rank: Invalid communicator, error stack:
    internal_Comm_rank(77): MPI_Comm_rank(comm=0x0, rank=0x580700) failed
    internal_Comm_rank(41): Invalid communicator

Not fully tested with current Piernik.

Not checked with late state of F40.