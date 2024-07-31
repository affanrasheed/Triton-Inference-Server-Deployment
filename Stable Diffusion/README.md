# Stable Diffusion Pipeline Using Implementation
In this tutorial, a stable diffusion pipeline is implemented using a Triton Inference Server. We will be using deeplearning models from different framework and use them together on a Triton Inference Server.
The flowchart for the complete implementation is given as
![stable diffusion flowchart](/img/multiple_backends.PNG)

# Requirement
## 1. Docker should be installed
[Docker installation guide](https://docs.docker.com/engine/install/)

# Running Instructions
## 1. Clone the Github Repository
## 2. Exporting models
In this part we will run NGC pytorch docker container to export the model and setup the model repository for Triton Inference Server. First go to the github folder, then run the pytorch container using the following command. 
```bash
# Replace yy.mm with year and month of release. Eg. 24.08
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:yy.mm-py3
```
Inside the container, we will install some dependencies. The CLIP text encoder and VAE decoder is exported to ONNX format and VAE Decoder is accerlated by converting to TRT format.
```bash
pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers==0.9.0
huggingface-cli login
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

# Place the models in the model repository
mkdir model_repository/vae/1
mkdir model_repository/text_encoder/1
mv vae.plan model_repository/vae/1/model.plan
mv encoder.onnx model_repository/text_encoder/1/model.onnx
```
## 3. Running the Model on Triton Inference Server
In this part, we will be running the triton inference server by running the container 
```bash
# Replace yy.mm with year and month of release. Eg. 24.08
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```
we will install some dependencies inside the container
```bash
# PyTorch & Transformers Lib
pip install torch torchvision torchaudio
pip install transformers ftfy scipy accelerate
pip install diffusers==0.9.0
pip install transformers[onnxruntime]
huggingface-cli login
```
Now we will lauch the Triton Inference Server
```bash
tritonserver --model-repository=/models
```
![server](/img/server.png)
## 4. Starting a GUI for sending Inference Request
In this part, we will be sending inference request to the Triton Inference Server and retrieve the results. First we will run the sdk container of triton on a separate terminal keeping the Triton Inference Server Running
```bash
# Replace yy.mm with year and month of release. Eg. 24.08
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash
```
we will install some dependencies inside the container
```bash
pip install packaging
pip install gradio==3.41.2
```
launch the gui 
```bash
python3 client.py --triton_url="localhost:8001"
```
## 5. Results
1. Input Prompt: Pikachu with a hat, 4k, 3d render
![result1](/img/res1.png)
2. Input Prompt: Realistic Human Sketch
![result2](/img/res2.png)
