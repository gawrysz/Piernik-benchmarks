#!/bin/bash

TEST="./benchmarking/standard_set_of_tests.sh"
DEST="../Piernik-benchmarks/cmp"

git co benchmarking
$TEST ${DEST}/b

git co benchmarking_master
$TEST ${DEST}/bm

echo " -d RIEMANN " >> .setuprc
$TEST ${DEST}/bmR

git co benchmarking_Rtransposed
$TEST ${DEST}/bmRt

sed -i 's/linear/weno3/' src/base/global.F90
$TEST ${DEST}/bmRtw

git co benchmarking_master
$TEST ${DEST}/bmRw

git co src/base/global.F90
sed -i '/RIEMANN/d' .setuprc
