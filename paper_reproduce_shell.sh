CUDA_VISIBLE_DEVICES=0 python main.py --dataset SVHN --model resnet18 --optimizer SGD --learning_rate 0.1 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=1 python main.py --dataset SVHN --model resnet18 --optimizer Adam --learning_rate 0.1 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=2 python main.py --dataset SVHN --model resnet18 --optimizer AdaBelief --learning_rate 0.1 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=3 python main.py --dataset SVHN --model resnet18 --optimizer SAM --learning_rate 0.1 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=4 python main.py --dataset SVHN --model resnet18 --optimizer SAM_Adam --learning_rate 0.1 --epochs 200 --train_verbose &

CUDA_VISIBLE_DEVICES=5 python main.py --dataset SVHN --model resnet18 --optimizer SGD --learning_rate 0.01 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=6 python main.py --dataset SVHN --model resnet18 --optimizer Adam --learning_rate 0.01 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=7 python main.py --dataset SVHN --model resnet18 --optimizer AdaBelief --learning_rate 0.01 --epochs 200 --train_verbose &
wait
CUDA_VISIBLE_DEVICES=0 python main.py --dataset SVHN --model resnet18 --optimizer SAM --learning_rate 0.01 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=1 python main.py --dataset SVHN --model resnet18 --optimizer SAM_Adam --learning_rate 0.01 --epochs 200 --train_verbose &

CUDA_VISIBLE_DEVICES=2 python main.py --dataset SVHN --model resnet18 --optimizer SGD --learning_rate 0.001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=3 python main.py --dataset SVHN --model resnet18 --optimizer Adam --learning_rate 0.001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=4 python main.py --dataset SVHN --model resnet18 --optimizer AdaBelief --learning_rate 0.001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=5 python main.py --dataset SVHN --model resnet18 --optimizer SAM --learning_rate 0.001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=6 python main.py --dataset SVHN --model resnet18 --optimizer SAM_Adam --learning_rate 0.001 --epochs 200 --train_verbose &

CUDA_VISIBLE_DEVICES=7 python main.py --dataset SVHN --model resnet18 --optimizer SGD --learning_rate 0.0001 --epochs 200 --train_verbose &
wait
CUDA_VISIBLE_DEVICES=0 python main.py --dataset SVHN --model resnet18 --optimizer Adam --learning_rate 0.0001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=1 python main.py --dataset SVHN --model resnet18 --optimizer AdaBelief --learning_rate 0.0001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=2 python main.py --dataset SVHN --model resnet18 --optimizer SAM --learning_rate 0.0001 --epochs 200 --train_verbose &
CUDA_VISIBLE_DEVICES=3 python main.py --dataset SVHN --model resnet18 --optimizer SAM_Adam --learning_rate 0.0001 --epochs 200 --train_verbose &
wait
#CUDA_VISIBLE_DEVICES=4 python main.py --dataset SVHN --model resnet18 --optimizer SGD --learning_rate 0.00001 --epochs 200 --train_verbose &
#CUDA_VISIBLE_DEVICES=5 python main.py --dataset SVHN --model resnet18 --optimizer Adam --learning_rate 0.00001 --epochs 200 --train_verbose &
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset SVHN --model resnet18 --optimizer AdaBelief --learning_rate 0.00001 --epochs 200 --train_verbose &
#CUDA_VISIBLE_DEVICES=7 python main.py --dataset SVHN --model resnet18 --optimizer SAM --learning_rate 0.00001 --epochs 200 --train_verbose &
#wait
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset SVHN --model resnet18 --optimizer SAM_Adam --learning_rate 0.00001 --epochs 200 --train_verbose &