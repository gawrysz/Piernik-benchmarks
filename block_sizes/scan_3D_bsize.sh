#!/bin/bash

d=96

for b in $( factor $(( $d /2 )) |\
	awk '{\
		for (i=2; i<=NF; i++)\
			a[$i]++;\
		n=0;\
		for (i in a) {\
			b[n]=i;\
			n++;\
		}\
		for (i0=0; i0<=a[b[0]]; i0++)\
			for (i1=0; i1<=a[b[1]]; i1++)\
				for (i2=0; i2<=a[b[2]]; i2++)\
					for (i3=0; i3<=a[b[3]]; i3++)\
						for (i4=0; i4<=a[b[4]]; i4++)\
							for (i5=0 ; i5<=a[b[5]]; i5++)\
								print 2 * b[0]**i0 * b[1]**i1 * b[2]**i2 * b[3]**i3 * b[4]**i4 * b[5]**i5;\
		}' | sort -nr | awk '{if ($1>=4) print}' ) ; do
	mpirun -np 1 ./piernik -n '$BASE_DOMAIN n_d = 3*'$d'/ $AMR bsize=3*'$b'/' 2> /dev/null |\
		grep dWallClock |\
		awk '{s+=$12} END {print '$d', '$b', s}'
done
