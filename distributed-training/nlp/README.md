<p align="center">
  <img src="../../assets/logo-classicblue-800px.png?raw=true" alt="Intel Logo" width="250"/>
</p>

# Intel® Cloud Optimization Modules for Microsoft Azure* 

© Copyright 2024, Intel Corporation

## nanoGPT Distributed Training

Intel® Optimized Cloud Modules for  Microsoft Azure* are a set of open source cloud-native reference architectures to facilitate building and deploying optimized, efficient, and scalable AI solutions on the Azure cloud. 

## Table of Contents
- [Introduction](#introduction)
- [Cloud Solution Architecture](#cloud-solution-architecture)
- [Prerequisites](#prerequisites)
- [Set up Azure Resources](#set-up-azure-resources)
- [Install Dependencies](#install-dependencies)
- [Fine-tune nanoGPT on a Single CPU](#fine-tune-nanogpt-on-a-single-cpu)
- [Prepare Environment for Distributed Fine-tuning](#prepare-environment-for-distributed-fine-tuning) 
- [Fine-tune nanoGPT on Multiple CPUs](#fine-tune-nanogpt-on-multiple-cpus)
- [Run Inference](#model-inference)
- [Clean up Resources](#clean-up-resources)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Large language models (LLMs) are becoming ubiquitous, but in many cases, you don’t need the full capability of the latest GPT model. Additionally, when you have a specific task at hand, the performance of the biggest GPT model may not be optimal. Often, fine-tuning a small LLM on your dataset can yield better results.  

The Intel® Cloud Optimization Modules for Microsoft Azure*: nanoGPT Distributed Training is designed to illustrate the process of fine-tuning a large language model with 3rd or 4th Generation [Intel® Xeon® Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html) on [Microsoft Azure](https://azure.microsoft.com/en-ca).  Specifically, we show the process of fine-tuning a nanoGPT model with 124M parameters on the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext) in a distributed architecture. This project builds upon the initial codebase of [nanoGPT](https://huggingface.co/datasets/Skylion007/openwebtext) built by Andrej Karpathy. The objective is to understand how to set up distributed training so that you can fine-tune the model to your specific workload. The result of this module will be a base LLM that can generate words, or tokens, that will be suitable for your use case when you modify it to your specific objective and dataset. 

This module demonstrates how to transform a standard single-node PyTorch training scenario into a high-performance distributed training scenario across multiple CPUs. To fully capitalize on Intel hardware and further optimize the fine-tuning process, this module integrates the [Intel® Extension for PyTorch*](https://intel.github.io/intel-extension-for-pytorch/) and [Intel® oneAPI Collective Communications Library (oneCCL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html). The module serves as a guide to setting up an Azure cluster for distributed training workloads while showcasing a complete project for fine-tuning LLMs

## Cloud Solution Architecture

To form the cluster, the cloud solution implements [Azure Trusted Virtual Machines](https://learn.microsoft.com/en-us/azure/virtual-machines/trusted-launch), leveraging instances from the [Dv5 series](https://learn.microsoft.com/en-us/azure/virtual-machines/dv5-dsv5-series). To enable seamless communication between the instances, each of the machines are connected to the same virtual network and a permissive network security group is established that allows all traffic from other nodes within the cluster. The raw dataset is taken from Hugging Face* Hub, and once the model has been trained, the weights are saved to the virtual machines. 

<p align="center">
  <img src="assets/architecture-diagram.png?raw=true" alt="Architecture Diagram"/>
</p>

[Back to Table of Contents](#table-of-contents)

## Prerequisites

This reference solution requires a [Microsoft Azure](https://azure.microsoft.com/en-ca)* account. 

Before getting started, download and install the [Microsoft Azure CLI](https://learn.microsoft.com/en-us/cli/azure/) version 2.46.0 or above for your operating system. To find your installation version, run:

```bash
az --version
```

## Set up Azure Resources

### I. Sign in with the Azure CLI

To begin setting up the Azure services for the module, first sign into your account:
```bash
az login
``` 

Alternatively, you can also use the [Azure Cloud Shell](https://learn.microsoft.com/en-us/azure/cloud-shell/overview), which automatically logs you in.

### II. Create an Azure Resource Group

We will first create an [Azure Resource Group](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/overview#resource-groups) to hold all of the related resources for our solution. The command below will create a resource group named `intel-nano-gpt` in the `eastus` region.

```bash
export RG=intel-nano-gpt
export LOC=eastus

az group create -n $RG -l $LOC
```

### III. Create an Azure Virtual Network

Next, we will create an [Azure virtual network](https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview) for the virtual machine cluster. This will allow the virtual machines (VMs) that we will create in the upcoming steps to securely communicate with each other.

The command below will create a virtual network named `intel-nano-gpt-vnet` and an associated subnet named `intel-nano-gpt-subnet`.

```bash
export VNET_NAME=intel-nano-gpt-vnet
export SUBNET_NAME=intel-nano-gpt-subnet

az network vnet create --name $VNET_NAME \
--resource-group $RG \
--address-prefix 10.0.0.0/16 \
--subnet-name $SUBNET_NAME \
--subnet-prefixes 10.0.0.0/24
```

### IV. Create an Azure Network Security Group

Now we will create an [Azure network security group](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview). This will filter the network traffic between the virtual machines in the network that we created in the previous step and allow us to SSH into the VMs in the cluster.

The following command will create an Azure network security group in our resource group named `intel-nano-gpt-network-security-group`.

```bash
export NETWORK_SECURITY_GROUP=intel-nano-gpt-network-security-group

az network nsg create -n $NETWORK_SECURITY_GROUP -g $RG
```

### V. Create an Azure SSH key pair

Next, we will create an [SSH key pair](https://learn.microsoft.com/en-us/azure/virtual-machines/ssh-keys-azure-cli) to securely connect to the Azure VMs in the cluster.
```bash
export SSH_KEY=intel-nano-gpt-SSH-key

az sshkey create -n $SSH_KEY -g $RG
```

Then, change privacy permissions of the private SSH key that was generated using the following command:
```bash
chmod 600 ~/.ssh/<private-ssh-key>
```

Next, upload the public SSH key to your Azure account in the resource group we created in the previous step.
```bash
az sshkey create -n $SSH_KEY -g $RG --public-key "@/home/<username>/.ssh/<ssh-key.pub>"
```

Verify the SSH key was uploaded successfully using:
```bash
az sshkey show -n $SSH_KEY -g $RG
```

### VI. Create an Azure Virtual Machine for Single-Node Training

Now we are ready to create the first [Azure virtual machine](https://azure.microsoft.com/en-ca/products/virtual-machines) in the cluster to fine-tune the model on a single node. We will create a virtual machine using an instance from the [`Dv5` Series](https://learn.microsoft.com/en-us/azure/virtual-machines/dv5-dsv5-series). This is a 3rd Generation Intel® Xeon® Scalable Processor featuring an all core turbo clock speed of 3.5 GHz with [Intel® Advanced Vector Extensions 512 (Intel® AVX-512)](https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html), [Intel® Turbo Boost Technology](https://www.intel.com/content/www/us/en/architecture-and-technology/turbo-boost/turbo-boost-technology.html), and [Intel® Deep Learning Boost](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/deep-learning-boost.html). These virtual machines offer a combination of vCPUs and memory to meet the requirements associated with most enterprise workloads, such as small-to-medium databases, low-to-medium traffic web servers, application servers, and more. For this module, we'll select the `Standard_D32_v5` VM size, which comes with 32 vCPUs and 128 GiB of memory.

```bash
export VM_NAME=intel-nano-gpt-vm
export VM_SIZE=Standard_D32_v5
export ADMIN_USERNAME=azureuser

az vm create -n $VM_NAME -g $RG \
--size $VM_SIZE \
--image Ubuntu2204 \
--os-disk-size-gb 256 \
--admin-username $ADMIN_USERNAME \
--ssh-key-name $SSH_KEY \
--public-ip-sku Standard \
--vnet-name $VNET_NAME \
--subnet $SUBNET_NAME \
--nsg $NETWORK_SECURITY_GROUP \
--nsg-rule SSH
```

Then, SSH into the VM using the following command:
```bash
ssh -i ~/.ssh/<private-ssh-key> azureuser@<public-IP-Address>
```

> **Note**: In the `~/.ssh` directory of your VM, ensure you have stored the private SSH key that was generated in [Step V](#v-create-an-azure-ssh-key-pair) in a file called `id_rsa` and have enabled the required privacy permissions using the command: `chmod 600 ~/.ssh/id_rsa`.

[Back to Table of Contents](#table-of-contents)

## Install Dependencies

You are now ready to set up the environment for fine-tuning the nanoGPT model. We will first update the package manager and install [tcmalloc](https://github.com/google/tcmalloc) for extra performance.

```bash
sudo apt update
sudo apt install libgoogle-perftools-dev unzip -y
```

Now let's set up a conda environment for fine-tuning GPT. First, download and install conda based on your operating system. You can find the download instructions [here](https://www.anaconda.com/download#downloads). The current commands for Linux are:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
bash ./Anaconda3-2023.07-1-Linux-x86_64.sh
```

To begin using conda, you have two options: restart the shell or execute the following command:
```bash
source ~/.bashrc
```

Running this command will source the `bashrc` file, which has the same effect as restarting the shell. This enables you to access and use conda for managing your Python environments and packages seamlessly.

Once conda is installed, create a virtual environment and activate it:

```bash
conda create -n cluster_env python=3.10
conda activate cluster_env
```

We have now prepared our environment and can move onto downloading data and training our nanoGPT model.

[Back to Table of Contents](#table-of-contents)

## Fine-tune nanoGPT on a Single CPU

Now that the Azure resources have been created, we'll download the OpenWebText data and fine-tune the model on a single node. 

First, clone the repo and install the dependencies:

```bash
git clone https://github.com/intel/intel-cloud-optimizations-azure
cd intel-cloud-optimizations-azure/distributed-training/nlp
pip install -r requirements.txt
```

In order to run distributed training, we will use the [Intel® oneAPI Collective Communications Library (oneCCL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html). Download the appropriate wheel file and install it using the following commands:

```bash
wget https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-1.13.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip install oneccl_bind_pt-1.13.0+cpu-cp310-cp310-linux_x86_64.whl
```

And you can delete the wheel file after installation:
```bash
rm oneccl_bind_pt-1.13.0+cpu-cp310-cp310-linux_x86_64.whl
```

Next, you can move onto downloading and processing the full OpenWebText dataset. This is all accomplished with one script.

```bash
python data/openwebtext/prepare.py --full
```

The complete dataset takes up approximately 54GB in the Hugging Face `.cache` directory and contains about 8 million documents (8,013,769). During the tokenization process, the storage usage might increase to around 120GB. The entire process can take anywhere from 30 minutes to 3 hours, depending on your CPU's performance.

Upon successful completion of the script, two files will be generated:

1. `train.bin`: This file will be approximately 17GB (~9B tokens) in size.
2. `val.bin`: This file will be around 8.5MB (~4M tokens) in size.

You should be able to run 
```bash
ls -ltrh data/openwebtext/
``` 
and see the following output:

<p align="center">
  <img src="assets/processed-data.png?raw=true" alt="Processed data"/>
</p>

To streamline the training process, we will use the [Hugging Face Accelerate library](https://huggingface.co/docs/accelerate/index). Once you have the processed `.bin` files, you are ready to generate the training configuration file by running the following accelerate command:

```bash
accelerate config --config_file ./single_config.yaml
```

When you run the above command, you will be prompted to answer a series of questions to configure the training process. Here's a step-by-step guide on how to proceed:

First, select `This machine` as we are not using Amazon SageMaker.
```bash
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
```

Next, since we are initially running the script on a single machine, select `No distributed training`. 

```bash
Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  No distributed training
    multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    TPU
```

You will be prompted to answer a few yes/no questions.  Here are the prompts and answers:
```bash
Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: NO
```

At the very end, you will be asked to select mixed precision. Select `bf16` on 4th Generation Intel Xeon CPUs; otherwise, you can select `fp16`.

```bash
Do you wish to use FP16 or BF16 (mixed precision)?
Please select a choice using the arrow or number keys, and selecting with enter
    no   
 ➔  fp16
    bf16
    fp8
```

This will generate a configuration file and save it as `single_config.yaml` in your current working directory.

We are now ready to start fine-tuning the nanoGPT model. To start the finetuning process, you can run the [`main.py`](main.py) script. But instead of running it directly, you can use the `accelerate launch` command along with the generated configuration file because `accelerate` automatically selects the appropriate number of cores, device, and mixed precision settings based on the configuration file, streamlining the process and optimizing performance. You can begin training at this point with:

```bash
accelerate launch --config_file ./single_config.yaml main.py
```

This command will initiate the fine-tuning process.

> **Note**: By default, [`main.py`](main.py) uses the [`intel_nano_gpt_train_cfg.yaml`](intel_nano_gpt_train_cfg.yaml) training configuration file: 

```yaml
data_dir: ./data/openwebtext

block_size: 1024
  
optimizer_config:
  learning_rate: 6e-4
  weight_decay: 1e-1
  beta1: 0.9
  beta2: 0.95

trainer_config:
  device: cpu
  mixed_precision: bf16          # fp16 or bf16
  eval_interval: 5               # how frequently to perform evaluation
  log_interval: 1                # how frequently to print logs
  eval_iters: 2                  # how many iterations to perform during evaluation
  eval_only: False
  batch_size: 32
  max_iters: 10                  # total iterations
  model_path: ckpt.pt 
  snapshot_path: snapshot.pt
  gradient_accumulation_steps: 2 
  grad_clip: 1.0
  decay_lr: True
  warmup_iters: 2
  lr_decay_iters: 10
  max_lr: 6e-4
  min_lr: 6e-5
  ```

You can review the file for `batch_size`, `device`, `max_iters`, etc. and make changes as needed. 

> **Note**: Accelerate by default will use the maximum number of physical cores (virtual cores excluded) by default. For experimental reasons, to control the number of threads, you can set `--num_cpu_threads_per_process` to the number of threads you wish to use. For example, if you want to run the script with only 4 threads:

```bash
accelerate launch --config_file ./single_config.yaml --num_cpu_threads_per_process 4 main.py
```

The script will train the model for a specified number of `max_iters` iterations and perform evaluations at regular `eval_interval`. If the evaluation score surpasses the previous model's performance, the current model will be saved in the current working directory under the name `ckpt.pt`. It will also save the snapshot of the training progress under the name `snapshot.pt`. You can easily customize these settings by modifying the values in the [`intel_nano_gpt_train_cfg.yaml`](intel_nano_gpt_train_cfg.yaml) file.

We performed 10 iterations of training, successfully completing the process. During this training, the model was trained on a total of 320 samples. This was achieved with a batch size of 32 and was completed in approximately seven and a half minutes.

<p align="center">
  <img src="assets/single-node-fine-tuning.png?raw=true" alt="Single node fine tuning"/>
</p>

The total dataset consists of approximately 8  million training samples, which would take much longer to train. However, the OpenWebText dataset was not built for a downstream task - it is meant to replicate the entire training dataset used for the base nanoGPT model. There are many smaller datasets like the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) (with 52K samples) that would be quite feasible on a distributed setup similar to the one described here.

> **Note**: In this fine-tuning process, we have opted not to use the standard PyTorch `DataLoader`. Instead, we have implemented a `get_batch` method that returns a batch of random samples from the dataset each time it is called. This implementation has been directly copied from the nanoGPT implementation. </br> Due to this specific implementation, we do not have the concept of epochs in the training process and instead are using iterations, where each iteration fetches a batch of random samples.

[Back to Table of Contents](#table-of-contents)

## Prepare Environment for Distributed Fine-tuning

Next, we need to prepare a new `accelerate` configuration file for multi-CPU setup. Before setting up the multi-CPU environment, ensure you have your machine's private IP address handy. To obtain it, run the following command:

```bash
hostname -i
```

With the private IP address ready, execute the following command to generate the new accelerate configuration file for the multi-CPU setup:
```bash
accelerate config --config_file ./multi_config.yaml
```

When configuring the multi-CPU setup using `accelerate config`, you will be prompted with several questions. To select the appropriate answers based on your environment, here is a step-by-step guide on how to proceed:

First, select `This machine` as we are not using Amazon SageMaker. 

```bash
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
```

Choose `multi-CPU` as the type of machine for our setup.

```bash
Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training       
 ➔  multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    TPU
```

Next, you can enter the number of instances you will be using. For example, here we have 3 (including the master node). 

```bash
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
```

Concerning the rank, since we are initially running this from the master node, enter `0`. For each machine, you will need to change the rank accordingly.

```bash
What is the rank of this machine?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  0
    1
    2
```

Next, you will need to provide the private IP address of the machine where you are running the `accelerate launch` command that we found earlier with `hostname -i`.

```bash
What is the IP address of the machine that will host the main process?   
```

Next, you can enter the port number to be used to communication. A commonly used port is `29500`, but you can choose any available port. 

```bash
What is the port you will use to communicate with the main process?   
```

The prompt of
```bash
How many CPU(s) should be used for distributed training?
```
is actually referring to the number of CPU sockets. Generally, each machine will have only 1 CPU socket. However, in the case of bare metal instances, you may have 2 CPU sockets per instance. Enter the appropriate number of sockets based on your instance configuration.

After completing the configuration, you will be ready to launch the multi-CPU fine-tuning process. The final output should look something like:

```plaintext
------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-CPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 3
------------------------------------------------------------------------------------------------------------------------------------------
What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? xxx.xxx.xxx.xxx
What is the port you will use to communicate with the main process? 29500
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: no
What rendezvous backend will you use? ('static', 'c10d', ...): static
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
How many CPU(s) should be used for distributed training? [1]:1
------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
fp16
```

This will generate a new configuration file named `multi_config.yaml` in your current working directory. Before creating a virtual machine image (VMI) from this volume, make sure to delete the `snapshot.pt` file. If this file exists, the `main.py` script will resume training from this snapshot.

```bash
rm snapshot.pt
```

### Create Two Additional Azure Virtual Machines for Distributed Fine-Tuning
Now we are ready to set up the additional VMs in the cluster for distributed fine-tuning. To ensure a consistent setup across the machines, we will create a virtual machine image (VMI) from the OS disk snapshot. This way we will not generalize the virtual machine currently running.

In a new terminal, create the [Azure snapshot](https://learn.microsoft.com/en-us/azure/virtual-machines/snapshot-copy-managed-disk) of the virtual machine's OS disk using the following command:

```bash
export DISK_NAME=intel-nano-gpt-disk-snapshot
export DISK_SOURCE=$(az vm show -n $VM_NAME -g $RG --query "storageProfile.osDisk.name" -o tsv)

az snapshot create -n $DISK_NAME -g $RG --source $DISK_SOURCE
```

Then, create an [Azure compute gallery](https://learn.microsoft.com/en-us/azure/virtual-machines/azure-compute-gallery) to store the virtual machine image definition and image version.

```bash
export GALLERY_NAME=intelnanogptgallery

az sig create -g $RG --gallery-name $GALLERY_NAME
```

Next, we will create the image definition for our VMI that will hold information about the image and requirements for using it. The image definition that we will create with the command below will be used to create a generalized Linux* image from the machine's OS disk.

```bash
export IMAGE_DEFINITION=intel-nano-gpt-image-definition

az sig image-definition create -g $RG \
--gallery-name $GALLERY_NAME \
--gallery-image-definition $IMAGE_DEFINITION \
--publisher Other --offer Other --sku Other \
--os-type linux --os-state Generalized
```

Now we are ready to create the image version using the disk snapshot and image definition we created above. This command may take a few moments to complete.

```bash
export IMAGE_VERSION=1.0.0
export OS_SNAPSHOT_ID=$(az snapshot show -g $RG -n $DISK_NAME --query "creationData.sourceResourceId" -o tsv)

az sig image-version create -g $RG \
--gallery-name $GALLERY_NAME \
--gallery-image-definition $IMAGE_DEFINITION \
--gallery-image-version $IMAGE_VERSION \
--os-snapshot $OS_SNAPSHOT_ID
```

Once the image version has been created, we can now create two additional virtual machines in our cluster using this version.

```bash
export VM_IMAGE_ID=$(az sig image-version show -g $RG --gallery-name $GALLERY_NAME --gallery-image-definition $IMAGE_DEFINITION --gallery-image-version $IMAGE_VERSION --query "id" -o tsv)

az vm create -n intel-nano-gpt-vms \
-g $RG \
--count 2 \
--size $VM_SIZE \
--image $VM_IMAGE_ID \
--admin-username $ADMIN_USERNAME \
--ssh-key-name $SSH_KEY \
--public-ip-sku Standard \
--vnet-name $VNET_NAME \
--subnet $SUBNET_NAME \
--nsg $NETWORK_SECURITY_GROUP \
--nsg-rule SSH
```

### Configure passwordless SSH

Next, with the private IP addresses of each of the nodes in the cluster, create an SSH configuration file located at `~/.ssh/config` on the master node. The configuration file should look like this:

```plaintext
Host 10.0.xx.xx
   StrictHostKeyChecking no

Host node1
    HostName 10.0.xx.xx
    User azureuser

Host node2
    HostName 10.0.xx.xx
    User azureuser
```

The `StrictHostKeyChecking no` line disables strict host key checking, allowing the master node to SSH into the worker nodes without prompting for verification.

With these settings, you can check your passwordless SSH by executing `ssh node1` or `ssh node2` to connect to any node without any additional prompts.

Next, on the master node, create a host file (`~/hosts`) that includes the names of all the nodes you want to include in the training process, as defined in the SSH configuration above. Use `localhost` for the master node itself as you will launch the training script from the master node. The `hosts` file will look like this:

```plaintext
localhost
node1
node2
```

This setup will allow you to seamlessly connect to any node in the cluster for distributed fine-tuning.

[Back to Table of Contents](#table-of-contents)

## Fine-tune nanoGPT on Multiple CPUs

Before beginning the fine-tuning process, it is important to update the `machine_rank` value on each machine. Follow these steps for each worker machine:

1. SSH into the worker machine.
2. Locate and open the `multi_config.yaml` file.
3. Update the value of the `machine_rank` variable in the file. Assign the rank to the worker nodes starting from 1.
   - For the master node, set the rank to 0.
   - For the first worker node, set the rank to 1.
   - For the second worker node, set the rank to 2.
   - Continue this pattern for additional worker nodes.

By updating the `machine_rank`, you ensure that each machine is correctly identified within the distributed fine-tuning setup. This is crucial for the successful execution of the fine-tuning process.

To fine-tune PyTorch models in a distributed setting on Intel hardware, we utilize the [Intel® oneAPI Collective Communications Library (oneCCL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html) and the [Intel® Message Passing Interface (Intel® MPI)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html). This implementation provides flexible, efficient, and scalable cluster messaging on Intel architecture. The [Intel® oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) includes all the necessary components, including `oneccl_bindings_for_pytorch`, which is installed alongside the MPI toolset.

Before launching the fine-tuning process, ensure you have set the environment variables for `oneccl_bindings_for_pytorch` in each node in the cluster by running the following command:

```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

This command sets up the environment variables required for utilizing `oneccl_bindings_for_pytorch` and enables distributed training using Intel MPI. 

> **Note**: In a distributed setting, `mpirun` can be used to run any program, not just for distributed training. It allows you to execute parallel applications across multiple nodes or machines, leveraging the capabilities of MPI (Message Passing Interface).

Finally, it's time to run the fine-tuning process on multi-CPU setup. The following command be used to launch distributed training:
```bash
mpirun -f ~/hosts -n 3 -ppn 1 -genv LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so" accelerate launch --config_file ./multi_config.yaml --num_cpu_threads_per_process 16 main.py 
```

Some notes on the arguments of `mpirun` to consider:
- `-n`: This parameter represents the number of CPUs or nodes. In our case, we specified `-n 3` to run on 3 nodes. Typically, it is set to the number of nodes you are using. However, in the case of bare metal instances with 2 CPU sockets per board, you would use `2n` to account for the 2 sockets.
- `-ppn`: The "process per node" parameter determines how many training jobs you want to start on each node. We only want 1 instance of each training to be run on each node, so we set this to `-ppn 1`. 
- `-genv`: This argument allows you to set an environment variable that will be applied to all processes. We used it to set the `LD_PRELOAD` environment variable to use the `libtcmaclloc` performance library.
- `-num_cpu_threads_per_process`: This argument specifies the number of CPU threads that PyTorch will use per process. We set this to use 16 threads in our case. When running deep learning tasks, it is best practice to use only the physical cores of your processor, which in our case is 16. 

Here is what the final output for distributed training would look like. 

<p align="center">
  <img src="assets/distributed-fine-tuning.png?raw=true" alt="Distributed fine tuning"/>
</p>

#### Author's note on the nanoGPT Implementation

The nanoGPT implementation utilizes a custom dataloader mechanism that randomly samples the designated dataset for fine-tuning. This approach results in a higher volume of data being sampled as the number of nodes grows in the distributed system. For instance, if you have a 10GB dataset and wish to perform distributed training on half of it (5GB), the dataloader can be configured to utilize 5GB/3 of data per node. The data is sampled randomly per node and tokenized all at once, saving time by bypassing iterative tokenization during each training batch. Consequently, the implementation does not have a concept of "epochs" but rather relies on steps to pass all of the selected data through the model.

[Back to Table of Contents](#table-of-contents)

## Model Inference

Now that we have fine-tuned the model, let's try to generate some text using the command below. 

```bash
python sample.py --ckpt_path=ckpt.pt
```

The script is designed to generate sample text containing 100 tokens. By default, the input prompt for generating these samples is the `It is interesting` prompt. However, you also have the option to specify your own prompt by using the `--prompt` argument as follows:

```bash
python sample.py --ckpt_path=ckpt.pt --prompt="This is new prompt"
```

Below is one sample generated text from the `It is interesting` input:

```
Input Prompt: It is interesting 
--------------- Generated Text ---------------
It is interesting  to see how many people like this, because I have been listening to and writing about this for a long time. 
Maybe I am just a fan of the idea of a blog, but I am talking about this particular kind of fan whose blog is about the other stuff I have like the work of Robert Kirkman and I am sure it is a fan of the work of Robert Kirkman. I thought that was really interesting and I am sure it can be something that I could relate to.

-------------------------------------------
```

This example does illustrate that the language model can generate text, but it is not useful in its current form until fine-tuned on downstream tasks. While there is repetition in the tokens here, this module's primary focus was on the successful distributed training process and leveraging the capabilities of Intel hardware effectively.

[Back to Table of Contents](#table-of-contents)

## Clean up Resources

When you are ready to delete all of the Azure resources and the resource group, run:

```
az group delete -n $RG --yes --no-wait
```

[Back to Table of Contents](#table-of-contents)

## Summary

By adopting distributed training techniques, we have achieved greater data processing efficiency. In approximately 6 minutes, we processed three times the amount of data as compared to non-distributed training methods. Additionally, we get a lower loss value indicating better model generalization. This performance boost and generalization enhancement is a testament to the advantages of leveraging distributed architectures for fine-tuning LLMs. 

The significance of distributed training in modern machine learning and deep learning scenarios lies in the following aspects:
- Faster training: As demonstrated in the output, distributed systems reduce the training time for large datasets. It allows parallel processing across multiple nodes, which accelerates the training process and enables efficient utilization of computing resources.
- Scalability: With distributed training, the model training process can easily scale to handle massive datasets, complex architectures, and larger batch sizes. This scalability is crucial for handling real-world, high-dimensional data.
- Model generalization: Distributed training enables access to diverse data samples from different nodes, leading to improved model generalization. This, in turn, enhances the model's ability to perform well on unseen data.

Overall, distributed training is an indispensable technique that empowers data scientists, researchers, and organizations to efficiently tackle complex machine learning tasks and achieve more performant results.

[Back to Table of Contents](#table-of-contents)

## Next Steps

- Learn more about all of the [Intel® Cloud Optimization Modules](https://www.intel.com/content/www/us/en/developer/topic-technology/cloud-optimization.html).
- Register for [Office Hours](https://software.seek.intel.com/SupportFromIntelExperts-Reg) for implementation support from Intel engineers. 
- Come chat with us on the [Intel® DevHub Discord server](https://discord.gg/rv2Gp55UJQ) to keep interacting with fellow developers.

[Back to Table of Contents](#table-of-contents)