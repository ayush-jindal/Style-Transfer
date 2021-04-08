#!/bin/bash

source ~/Py37/bin/activate
cd ~/Desktop/Python/Papers/Style\ Transfer/trial\ builds/src\ code\ 2

python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 1e3 -b 1e1 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 1e3 -b 1e3 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 1e3 -b 1e5 --optimizer sgd --max-size 224 -v

python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 3e3 -b 1e1 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 3e3 -b 1e3 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 3e3 -b 1e5 --optimizer sgd --max-size 224 -v

python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 5e3 -b 1e1 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 5e3 -b 1e3 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 5e3 -b 1e5 --optimizer sgd --max-size 224 -v

python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 7e3 -b 1e1 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 7e3 -b 1e3 --optimizer sgd --max-size 224 -v
python driver.py -c ./../golden_gate.jpg -s ./../candy.jpg -e 7e3 -b 1e5 --optimizer sgd --max-size 224 -v

deactivate
