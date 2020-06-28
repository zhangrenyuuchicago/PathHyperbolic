for t in `seq 0 3`;
do
    for fold in `seq 0 4` ;
    do
        echo "current t: ${t} fold: ${fold}"
        CUDA_VISIBLE_DEVICES=0 python main.py -f ${fold}
    done
done
