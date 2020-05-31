


export DATE=20200504
export VERSION=5


#for CNT in {1..2}; do
#    #export OUTPUT="date${DATE}_v${VERSION}_CNT${CNT}"
#    export EXTRA_ARGS=" --output-dir output/date${DATE}_v${VERSION}_CNT${CNT}/"
#    export SBATCH_JOB_NAME="codi_date${DATE}_v${VERSION}_CNT${CNT}"
#    sbatch < ml_batch_script.sh
#done



for NEPOCHS in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200; do
    #export OUTPUT="date${DATE}_v${VERSION}_NEPOCH${NEPOCHS}"
    export EXTRA_ARGS=" --output-dir output/date${DATE}_v${VERSION}_NEPOCH${NEPOCHS}/ --n-epochs ${NEPOCHS}"
    export SBATCH_JOB_NAME="codi_date${DATE}_v${VERSION}_NEPOCH${NEPOCHS}"
    sbatch < ml_batch_script.sh
done
