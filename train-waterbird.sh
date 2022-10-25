# ERM
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity


# DBAT: D_ood = D_test
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
python ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet50 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
