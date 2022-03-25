export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LC_ALL=C

./Tools/conlleval < ./Data/Conll2003_BMES/test.txt.shuffle.out 


