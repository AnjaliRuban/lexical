#!/bin/bash

runner=${1}
types=${2}

unk=100

while [ $unk -le 101 ]
	do
		bash ${runner} ${types} unk${unk} 0.${unk}
		unk=$(($unk + 10))
	done

