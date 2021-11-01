#!/bin/bash

runner=${1}
types=${2}

unk=60

while [ $unk -le 100 ]
	do
		bash ${runner} ${types} unk${unk} 0.${unk}
		unk=$(($unk + 10))
	done

