Fedora 41 is usable for current Piernik, hard to perform benchmarking set of tests.

* openmpi-5.0.5-2.fc41 works only for some configurations. It can grind
  the standard benchmarking set only with df5206cc9 patch (like in F40).
  It crashes a lot on strong scaling otherwise.

* mpich-4.2.2-1.fc41 doesn't work witch benchmarking branch anymore, even
  single thread doesn't run because

    Abort(138071315) on node 0: Fatal error in internal_Wait: Invalid MPI_Request, error stack:
    internal_Wait(68205): MPI_Wait(request=0x7ffcb6f37f28, status=0x1) failed
    internal_Wait(68146): Invalid MPI_Request

Both MPI libraries work with PR #582.
