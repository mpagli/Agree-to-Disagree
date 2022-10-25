# ERM
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 --no_diversity


# D-BAT
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0.00001 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0.00001 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 600 --lr 0.001 --opt sgd --scheduler none --alpha 0.00001 --perturb_type ood_is_not_test --model resnet18 --dataset oh-65cls --eval_freq 1000 --l2_reg 0.0001 
