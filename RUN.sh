# Multiple run script

for j in {1..5};
do
python main.py exec --mode='test' --run=$j --amount=10  --emb='de'
done

for j in {6..10};
do
python main.py exec --mode='test' --run=$j --amount=10  --emb='mul'
done
