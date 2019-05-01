#!/bin/bash

TEST="./benchmarking/standard_set_of_tests.sh"
DEST="../Piernik-benchmarks/cmp"

RC=".setuprc"
RCSAVE=${RC}".save"

cp $RC $RCSAVE

echo benchmarking_RTVD
git co benchmarking
$TEST ${DEST}/benchmarking_RTVD

for b in bench_fnostack_arrays bench_fstack_arrays ; do
    for s in RTVD RIEMANN HLLC ; do
    	echo ${b}_${s}
	git co $b
	cp $RCSAVE $RC
	echo " -d "$s >> .setuprc
	$TEST ${DEST}/${b}_${s}
    done
done

cp $RCSAVE $RC
