#BSUB -J "jupyter notebook"
#BSUB -n 4
#BSUB -q prod.med
#BSUB -o "./jn.OUT"
#BSUB -e "./jn.ERR"
#BSUB -R "select[ngpus>0] rusage [ngpus_excl_p=1]"

source activate torch36
jupyter notebook --port 6969 --no-browser
