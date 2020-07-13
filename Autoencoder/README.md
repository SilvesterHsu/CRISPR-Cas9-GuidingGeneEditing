# Document
## 1. Environment
The entire project is built on a deep learning framework. To facilitate exploration, we used **pytorch**.
### Docker Environment
Our development environment also provides Docker images [docker-pytorch](https://hub.docker.com/r/silvesterhsu/docker-pytorch). 

> If you already have a pytorch development environment, you can skip environment section.

[Docker](https://www.docker.com/) is a virtual environment that can easily create, delete, download, and upload any development environment. Therefore, you can also use Docker to pull the development environment of this project without installing any basic software support. 

**Step 1: Install Docker** 
Run the code in a terminal.
```shell
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```
If you want more [detail](https://docs.docker.com/engine/install/ubuntu/).


**Step 2: Add GPU Support (optional)**
At the same time, this [image](https://hub.docker.com/r/silvesterhsu/docker-pytorch) supports GPU acceleration. 
```shell
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
For more [detail](https://github.com/NVIDIA/nvidia-docker).
> Note that due to Docker virtualization, GPU acceleration is only available under Linux systems.

**Step 3: Pull Image**
For CPU user:
```shell
docker run --restart=always --name pytorch -ti --ipc=host -p "$PORT":8888 -v "$PWD":/workspace silvesterhsu/docker-pytorch
```

For GPU user:
```shell
docker run --gpus all --restart=always --name pytorch -ti --ipc=host -p "$PORT":8888 -v "$PWD":/workspace silvesterhsu/docker-pytorch
```

`$PORT`: Port mapping. It is the port that needs to link the local to the image. In docker, jupyter will open port `8888` as a web access. If the local port `8888` is not occupied, it is recommended to use `8888`.

`$PWD`: File mapping. Project work path

**Please** follow the instruction in our [Docker Image](https://hub.docker.com/repository/docker/silvesterhsu/docker-pytorch).

## 2. Usage
Just open the jupyter file and run it directly.