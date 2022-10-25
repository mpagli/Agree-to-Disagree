# ERM
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.0 --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.0 --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.0 --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001 --no_diversity


# DBAT: D_ood = D_test
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001


# DBAT: D_ood != D_test
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_not_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_not_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 2 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 60 --lr 0.001 --opt sgd --scheduler none --alpha 0.000001 --perturb_type ood_is_not_test --model resnet50 --dataset camelyon17 --eval_freq 5000 --l2_reg 0.0001
