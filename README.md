# Triton Server Segmentation Fault With Custom TensorRT Plugin

This repo reproduces the segmentation fault when launching nvidia triton server with custom tensorrt plugin.  

It uses directly the detection layer code from https://github.com/NVIDIA/TensorRT/tree/master/plugin/detectionLayerPlugin to build a .so custom plugin. Lauching triton server with the built pluggin result segmentation error.


## Environment 
I am using AWS g4dn.xlarge instance, it use T4 GPU.

To build the plugin I use directly the latest NGC TensorRT container nvcr.io/nvidia/tensorrt:20.10-py3

To launch the triton server I use the latest NGC Triton Server container nvcr.io/nvidia/tritonserver:20.10-py3

## Build the plugin 

Within the Tensorrt container, standard procedure used to build the plugin
```
mkdir build
cd build
cmake ..
make
```
The resulted plugin `libtestplugins.so` is created

## Launch triton server
Within the Triton server container, assuming the model is in path `/ubuntu/model_repository` and the plugin is at `/ubuntu/libtestplugins.so`:
```
LD_PRELOAD=/ubuntu/libtestplugins.so tritonserver --model-repository=/ubuntu/model_repository --strict-model-config=false
```

The produced error messages are:
```
I1105 06:27:12.184359 1991 metrics.cc:184] found 1 GPUs supporting NVML metrics
I1105 06:27:12.189796 1991 metrics.cc:193]   GPU 0: Tesla T4
I1105 06:27:12.362048 1991 pinned_memory_manager.cc:195] Pinned memory pool is created at '0x7fd970000000' with size 268435456
I1105 06:27:12.362442 1991 cuda_memory_manager.cc:98] CUDA memory pool is created on device 0 with size 67108864
Segmentation fault (core dumped)
```

