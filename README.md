# Multi-time-step Segment Routing based TrafficEngineering Leveraging Traffic Prediction

This is the implementation for the paper "Multi-time-step Segment Routing based TrafficEngineering Leveraging Traffic
Prediction" - under review at IM2021.

# Training gwn

python train.py --do_graph_conv {--aptonly --addaptadj} --type {p1/p2/p3}

python train.py --do_graph_conv --aptonly --addaptadj --randomadj --train_batch_size 16 --val_batch_size 16 --dataset
abilene_tm --test --run_te
