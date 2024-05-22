export NO_PLT_SHOW=False

for i in {0..10}
do
  echo $i
  export TRAIN_RANGE=$i
  export TRAIN_NAME=NIHAT$i 
  python run.py
done

