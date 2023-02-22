#!/bin/bash

mkdir experiments

wget https://www.dropbox.com/s/mbbae2yk7cexwig/MOViA.zip?dl=1
unzip MOViA.zip?dl=1
rm MOViA.zip?dl=1
rsync -va --delete-after MOViA experiments
rm -r MOViA

wget https://www.dropbox.com/s/luuzfmo3v2ka2kb/Obj3D.zip?dl=1
unzip Obj3D.zip?dl=1
rm Obj3D.zip?dl=1
rsync -va --delete-after Obj3D experiments
rm -r Obj3D
