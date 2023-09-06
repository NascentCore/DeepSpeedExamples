# DeepSpeed on K8s

Use Bert training written in PyTorch+DeepSpeed, used to test 3k.

## Training code

```
# Create conda environment: bert
conda create -n bert python==3.11
conda activate bert

git clone git@github.com:NascentCore/3k.git
cd 3k/examples/bert

apt install libopenmpi-dev
pip install -r requirements.txt

# Run 1 GPU locally, no deepspeed distributed.
python train_bert.py  --local_rank 0 --checkpoint_dir ./experiments

# Use all GPUs on localhost
deepspeed train_bert_ds.py --checkpoint_dir ./experiments

# Enable 0,2,3 GPU
CUDA_VISIBLE_DEVICES=0,2,3 deepspeed train_bert_ds.py --checkpoint_dir ./experiments

# Distributed training
# 要求：配置hostfile文件；每台机器上代码存储路径都一致，
# 并且对应的conda虚拟环境也一致;各台机器间ssh免密登陆；有.deepspeed_env文件；
# 注：当在多个节点上进行训练时，我们发现支持传播用户定义的环境变量非常有用。
# 默认情况下，DeepSpeed 将传播所有设置的 NCCL 和 PYTHON 相关环境变量。
# 如果您想传播其它变量，可以在名为 .deepspeed_env 的文件中指定它们，
# 该文件包含一个行分隔的 VAR=VAL 条目列表。
# DeepSpeed 启动器将查找你执行的本地路径以及你的主目录（~/）
deepspeed --hostfile=hostfile  --master_port 60000 train_bert_ds.py \
    --checkpoint_dir ./experiments
```

## Container

When the training PyTorch code is updated, run the following to build and push
new image version.

```
docker build . -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert:<version>
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert:<version>
```

If you need to update the `bert-base` image:

```
docker build . -f base.Dockerfile -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert-base
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert-base
```
