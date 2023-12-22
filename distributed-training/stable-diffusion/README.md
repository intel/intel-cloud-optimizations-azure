<p align="center">
  <img src="../../assets/logo-classicblue-800px.png?raw=true" alt="Intel Logo" width="250"/>
</p>

# Intel® Optimized Cloud Modules for Microsoft Azure* 

© Copyright 2023, Intel Corporation

## Stable Diffusion Distributed Training

Intel® Optimized Cloud Modules for  Microsoft Azure* are a set of open source cloud-native reference architectures to facilitate building and deploying optimized, efficient, and scalable AI solutions on the Azure cloud.

## Table of Contents

- [Introduction](#introduction)
- [Cloud Solution Architecture](#cloud-solution-architecture)
- [Prerequisites](#prerequisites)
- [Set up Azure Resources](#set-up-azure-resources)
- [Install Dependencies](#install-dependencies)
- [Fine-tune on a Single CPU](#fine-tune-on-a-single-cpu)
- [Prepare Environment for Distributed Fine-tuning](#prepare-environment-for-distributed-fine-tuning)
- [Fine-tune on Multiple CPUs](#fine-tune-on-multiple-cpus)
- [Clean up Resources](#clean-up-resources)
- [Next Steps](#next-steps)

## Introduction

A Stable Diffusion Generative Text-to-Image Model leverages a diffusion process to convert textual descriptions into coherent image representations, establishing a robust foundation for multimodal learning tasks. Fine-tuning this model enables users to tailor its generative capabilities towards specific domains or datasets, thereby improving the quality and relevance of the produced imagery.

As the fine-tuning process can be computationally intensive, especially with burgeoning datasets, distributing the training across multiple [Intel® Xeon® Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html) combined with the [Intel® Extension for PyTorch*](https://intel.github.io/intel-extension-for-pytorch/) can significantly accelerate the fine-tuning task.

This [Intel® Optimized Cloud Module](https://www.intel.com/content/www/us/en/developer/topic-technology/cloud-optimization.html) is focused on providing instructions for executing this fine-tuning workload on the Azure cloud using 3rd or 4th Generation Intel Xeon Processors and software optimizations offered through the Hugging Face* Accelerate library.

## Cloud Solution Architecture

To form the cluster, the cloud solution implements [Azure Trusted Virtual Machines](https://learn.microsoft.com/en-us/azure/virtual-machines/trusted-launch), leveraging instances from the [Dv5 series](https://learn.microsoft.com/en-us/azure/virtual-machines/dv5-dsv5-series), which are powered by 3rd Generation Intel® Xeon® Scalable Processors. To enable seamless communication between the instances, each of the machines are connected to the same virtual network and a permissive network security group is established that allows all traffic from other nodes within the cluster. The raw images and diffusion model are downloaded from the Hugging Face Hub, and once the model has been trained, the weights are saved to the virtual machines. 

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

We will first create an [Azure Resource Group](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/overview#resource-groups) to hold all of the related resources for our solution. The command below will create a resource group named `intel-stable-diffusion` in the `eastus` region.

```bash
export RG=intel-stable-diffusion
export LOC=eastus

az group create -n $RG -l $LOC
```

### III. Create an Azure Virtual Network

Next, we will create an [Azure virtual network](https://learn.microsoft.com/en-us/azure/virtual-network/virtual-networks-overview) for the cluster. This will allow the virtual machines (VMs) that we will create in the upcoming steps to securely communicate with each other.

The command below will create a virtual network named `intel-stable-diffusion-vnet` and an associated subnet named `intel-stable-diffusion-subnet`.

```bash
export VNET_NAME=intel-stable-diffusion-vnet
export SUBNET_NAME=intel-stable-diffusion-subnet

az network vnet create --name $VNET_NAME \
--resource-group $RG \
--address-prefix 10.0.0.0/16 \
--subnet-name $SUBNET_NAME \
--subnet-prefixes 10.0.0.0/24
```

### IV. Create an Azure Network Security Group

Now we will create an [Azure network security group](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview). This will filter the network traffic between the virtual machines in the network that we created in the previous step and allow us to SSH into the VMs in the cluster.

The following command will create an Azure network security group in our resource group named `intel-stable-diffusion-network-security-group`.

```bash
export NETWORK_SECURITY_GROUP=intel-stable-diffusion-network-security-group

az network nsg create -n $NETWORK_SECURITY_GROUP -g $RG
```

### V. Create an Azure SSH key pair

Next, we will create an [SSH key pair](https://learn.microsoft.com/en-us/azure/virtual-machines/ssh-keys-azure-cli) to securely connect to the Azure VMs in the cluster.
```bash
export SSH_KEY=intel-stable-diffusion-SSH-key

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

Now we are ready to create the first [Azure virtual machine](https://azure.microsoft.com/en-ca/products/virtual-machines) in the cluster to fine-tune the model on a single node. We will create a virtual machine using an instance from the [`Dv5` Series](https://learn.microsoft.com/en-us/azure/virtual-machines/dv5-dsv5-series). Azure Dv5 instances are powered by 3rd Generation Intel® Xeon® Scalable Processors featuring an all core turbo clock speed of 3.5 GHz with [Intel® Advanced Vector Extensions 512 (Intel® AVX-512)](https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html), [Intel® Turbo Boost Technology](https://www.intel.com/content/www/us/en/architecture-and-technology/turbo-boost/turbo-boost-technology.html), and [Intel® Deep Learning Boost](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/deep-learning-boost.html). These virtual machines offer a combination of vCPUs and memory to meet the requirements associated with most enterprise workloads, such as small-to-medium databases, low-to-medium traffic web servers, application servers, and more. For this module, we'll select the `Standard_D8_v5` VM size, which comes with 8 vCPUs and 32 GiB of memory.

```bash
export VM_NAME=intel-stable-diffusion-vm
export VM_SIZE=Standard_D8_v5
export ADMIN_USERNAME=azureuser

az vm create -n $VM_NAME -g $RG \
--size $VM_SIZE \
--image Ubuntu2204 \
--os-disk-size-gb 32 \
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

> **Note**: In the `~/.ssh` directory of your machine, ensure you have stored the private SSH key that was generated in [Step V](#v-create-an-azure-ssh-key-pair) in a file called `id_rsa` and have enabled the required privacy permissions using the command: `chmod 600 ~/.ssh/id_rsa`.

[Back to Table of Contents](#table-of-contents)

## Install Dependencies

You are now ready to set up the environment for stable diffusion. We will first update the package manager and install `make`.

```bash
sudo apt update
sudo apt install make
```

Then, clone the repository and execute the `makefile` script in the `stable-diffusion` directory to set up the environment on the master node.

```bash
git clone https://github.com/intel/intel-cloud-optimizations-azure.git
cd intel-cloud-optimizations-azure/distributed-training/stable-diffusion
make stable-diffusion-env
```

This script will set up all of the dependencies for the module. These include:
- Setting up a `conda` environment
- Installing `PyTorch`, `Intel Extension for PyTorch`, `oneCCL` bindings, `transformers`, and `accelerate`

Upon successful completion of the `stable-diffusion-env` make target, you should see a new directory called `dicoo`. This directory will hold the training images for the stable diffusion model, `0.jpeg` - `5.jpeg`, approximately 288KiB in size.

[Back to Table of Contents](#table-of-contents)

## Fine-tune on a Single CPU

To optimize the fine-tuning process, we will use the [Intel® Extension for PyTorch*](https://intel.github.io/intel-extension-for-pytorch/). The Intel Extension for PyTorch elevates PyTorch performance on Intel hardware with the integration of the newest features and optimizations that have not yet been incorporated into the open source distribution. This extension efficiently utilizes Intel hardware capabilities, such as [Intel® Advanced Matrix Extensions (Intel® AMX)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) and [Intel® Advanced Vector Extensions 512 (Intel® AVX-512)](https://www.intel.com/content/www/us/en/architecture-and-technology/avx-512-overview.html) instruction sets on Intel Xeon CPUs.

The optimizations for PyTorch have been added to the `textual_inversion.py` script included in this repository. To enable the accelerations, simply apply the `optimize` function to the model object shown in the code snippet below:

```python
unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)

import intel_extension_for_pytorch as ipex
unet = ipex.optimize(unet, dtype=weight_dtype)
vae = ipex.optimize(vae, dtype=weight_dtype) 
```

To test on a single node, we can run the `single-node-fine-tuning` make target.

```bash
make single-node-fine-tuning
```

The single node fine-tuning will run five steps and save the `learned_embeds.safetensors` in the `textual_inversion_output` directory once completed.

Before preparing the Azure resources for distributed fine-tuning, first delete the `textual_inversion_output` directory by running:

```bash
rm -rf textual_inversion_output
```

[Back to Table of Contents](#table-of-contents)

## Prepare Environment for Distributed Fine-tuning

Now that we have successfully run the stable diffusion model on a single node, we will begin preparing our environment for distributed fine-tuning. To streamline the fine-tuning process in a distributed setting, we will use the Hugging Face* [Accelerate](https://huggingface.co/docs/accelerate/index) library. 

Before generating the `accelerate` configuration file, first obtain your machine's private IP address by executing the following command:

```bash
hostname -i
```

With the private IP address ready, execute the following command to generate the new accelerate configuration file for the multi-CPU setup:
```bash
accelerate config
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
How many different machines will you use (use more than 1 for multi-node training)? [1]: 3
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

This will generate a new configuration file named `default_config.yaml` in the `~/.cache/huggingface/accelerate` directory.

### Create Two Additional Azure Virtual Machines for Distributed Fine-Tuning

To set up the additional VMs in the cluster, we will create a virtual machine image from the OS disk snapshot. This way we will not generalize the virtual machine currently running.

In a new terminal, we'll first create the [Azure snapshot](https://learn.microsoft.com/en-us/azure/virtual-machines/snapshot-copy-managed-disk) of our virtual machine's OS disk using the following command:

```bash
export DISK_NAME=intel-stable-diffusion-disk-snapshot
export DISK_SOURCE=$(az vm show -n $VM_NAME -g $RG --query "storageProfile.osDisk.name" -o tsv)

az snapshot create -n $DISK_NAME -g $RG --source $DISK_SOURCE
```

Then, we will create an [Azure compute gallery](https://learn.microsoft.com/en-us/azure/virtual-machines/azure-compute-gallery) to store the virtual machine image definition and image version.

```bash
export GALLERY_NAME=intelstablediffusiongallery

az sig create -g $RG --gallery-name $GALLERY_NAME
```

Next, we will create the image definition for our VMI that will hold information about the image and requirements for using it. The image definition that we will create with the command below will be used to create a generalized Linux* image from our machine's OS disk.

```bash
export IMAGE_DEFINITION=intel-stable-diffusion-image-definition

az sig image-definition create -g $RG \
--gallery-name $GALLERY_NAME \
--gallery-image-definition $IMAGE_DEFINITION \
--publisher Other --offer Other --sku Other \
--os-type linux --os-state Generalized
```

Now we will create the image version using the disk snapshot and image definition we created above. This command may take a few moments to complete.

```bash
export IMAGE_VERSION=1.0.0
export OS_SNAPSHOT_ID=$(az snapshot show -g $RG -n $DISK_NAME --query "creationData.sourceResourceId" -o tsv)

az sig image-version create -g $RG \
--gallery-name $GALLERY_NAME \
--gallery-image-definition $IMAGE_DEFINITION \
--gallery-image-version $IMAGE_VERSION \
--os-snapshot $OS_SNAPSHOT_ID
```

Once the image version has been created, we can now create two additional virtual machines in our cluster using this image.

```bash
export VM_IMAGE_ID=$(az sig image-version show -g $RG --gallery-name $GALLERY_NAME --gallery-image-definition $IMAGE_DEFINITION --gallery-image-version $IMAGE_VERSION --query "id" -o tsv)

az vm create -n intel-stable-diffusion-vms \
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

Next, with the private IP addresses of each of the VMs in the cluster, create an SSH configuration file located at `~/.ssh/config` on the master node. The configuration file should look like this:

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

## Fine-Tune on Multiple CPUs

Before beginning the distributed fine-tuning process, it is important to update the `machine_rank` value on each machine. Follow these steps for each worker machine:

1. SSH into the worker machine.
2. Locate and open the `default_config.yaml` in the `~/.cache/huggingface/accelerate` directory.
3. Update the value of the `machine_rank` variable in the file. Assign the rank to the worker nodes starting from 1.
   - For the master node, set the rank to 0.
   - For the first worker node, set the rank to 1.
   - For the second worker node, set the rank to 2.
   - Continue this pattern for additional worker nodes.

By updating the `machine_rank`, you ensure that each machine is correctly identified within the distributed fine-tuning setup. This is crucial for the successful execution of the fine-tuning process.

To fine-tune PyTorch models in a distributed setting on Intel hardware, we utilize the [Intel® oneAPI Collective Communications Library (oneCCL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html) and the [Intel® Message Passing Interface (Intel® MPI)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html). This implementation provides flexible, efficient, and scalable cluster messaging on Intel architecture. The [Intel® oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) includes all the necessary components, including `oneccl_bindings_for_pytorch`, which is installed alongside the MPI toolset.

Before launching the fine-tuning process, ensure you have set the environment variables for `oneccl_bindings_for_pytorch` in each node in the cluster by executing the following commands:

```bash
conda activate stable_diffusion
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

This command sets up the environment variables required for utilizing `oneccl_bindings_for_pytorch` and enables distributed training using Intel MPI. 

> **Note**: In a distributed setting, `mpirun` can be used to run any program, not just for distributed training. It allows you to execute parallel applications across multiple nodes or machines, leveraging the capabilities of MPI (Message Passing Interface).

Finally, it's time to run the fine-tuning process on multi-CPU setup. The following command can be used to launch the distributed fine-tuning process:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="dicoo"

mpirun -f ~/hosts -n 3 -ppn 1 accelerate launch textual_inversion.py --pretrained_model_name_or_path=$MODEL_NAME --train_data_dir=$DATA_DIR --learnable_property="object"   --placeholder_token="<dicoo>" --initializer_token="toy" --resolution=512  --train_batch_size=1  --seed=7  --gradient_accumulation_steps=1 --max_train_steps=30 --learning_rate=2.0e-03 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir=./textual_inversion_output --mixed_precision bf16 --save_as_full_pipeline
```

Some notes on the arguments for `mpirun` to consider:
- `-n`: This parameter represents the number of CPUs or nodes. In our case, we specified `-n 3` to run on 3 nodes. Typically, it is set to the number of nodes you are using. However, in the case of bare metal instances with 2 CPU sockets per board, you would use `2n` to account for the 2 sockets.
- `-ppn`: The "process per node" parameter determines how many training jobs you want to start on each node. We only want 1 instance of each training to be run on each node, so we set this to `-ppn 1`. 
- `--pretrained_model_name_or_path`: Path to pretrained model or model identifier from [huggingface.co/models](https://huggingface.co/models).
- `--learnable_property`: Choose between `object` and `style`.
- `--placeholder_token`: A token to use as a placeholder for the concept.
- `--initializer_token`: A token to use as initializer word.
- `--resolution`: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
- `--train_batch_size`: Batch size (per device) for the training dataloader.
- `--seed`: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
- `--gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
- `--max_train_steps`: Total number of training steps to perform. If provided, overrides `num_train_epochs`.
- `--learning_rate`: Initial learning rate (after the potential warmup period) to use.
- `--lr_scheduler`: The scheduler type to use. Choose between `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`.
- `--lr_warmup_steps`: Number of steps for the warmup in the `lr_scheduler`.
- `--output_dir`: The output directory where the model predictions and checkpoints will be written.
- `--mixed_precision`: Whether to use mixed precision. Choose between `bf16` (bfloat16) and `fp16`. To use `bf16` requires PyTorch >= 1.10 and an Nvidia Ampere GPU.
- `--save_as_full_pipeline`: Save the complete stable diffusion pipeline.

[Back to Table of Contents](#table-of-contents)

## Clean up Resources

When you are ready to delete all of the resources and the resource group, run:

```
az group delete -n $RG --yes --no-wait
```

[Back to Table of Contents](#table-of-contents)

## Next Steps

- Learn more about all of the [Intel® Cloud Optimization Modules](https://www.intel.com/content/www/us/en/developer/topic-technology/cloud-optimization.html).
- Register for [Office Hours](https://software.seek.intel.com/SupportFromIntelExperts-Reg) for help with your implementation. 
- Come chat with us on the [Intel® DevHub Discord server](https://discord.gg/rv2Gp55UJQ) to keep interacting with fellow developers.

[Back to Table of Contents](#table-of-contents)