## [NVIDIA CUDA 工具包发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#abstract)

CUDA 工具包的发行说明。



## [1. CUDA 11.4 发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#title-new-features)

NVIDIA® CUDA® Toolkit 的发行说明可以在http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html在线找到。

**注意：**发行说明已重新组织为两个主要部分：一般 CUDA 发行说明和包含 11.x 版本历史信息的 CUDA 库发行说明。



### [1.1. CUDA 工具包主要组件版本](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)

- CUDA 组件

  从 CUDA 11 开始，工具包中的各种组件都是独立的版本。对于 CUDA 11。4、版本如下表所示：

  | Component Name                                               | Version Information | Supported Architectures         |
  | ------------------------------------------------------------ | ------------------- | ------------------------------- |
  | CUDA Runtime (cudart)                                        | 11.4.100            | x86_64, POWER, Arm64            |
  | cuobjdump                                                    | 11.4.43             | x86_64, POWER, Arm64            |
  | CUPTI                                                        | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA cuxxfilt (demangler)                                    | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA Demo Suite                                              | 11.4.100            | x86_64                          |
  | CUDA GDB                                                     | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA Memcheck                                                | 11.4.100            | x86_64, POWER                   |
  | CUDA NVCC                                                    | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA nvdisasm                                                | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA NVML Headers                                            | 11.4.43             | x86_64, POWER, Arm64            |
  | CUDA nvprof                                                  | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA nvprune                                                 | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA NVRTC                                                   | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA NVTX                                                    | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA NVVP                                                    | 11.4.100            | x86_64, POWER                   |
  | CUDA Samples                                                 | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA Compute Sanitizer API                                   | 11.4.100            | x86_64, POWER, Arm64            |
  | CUDA Thrust                                                  | 11.4.43             | x86_64, POWER, Arm64            |
  | CUDA cuBLAS                                                  | 11.5.4.8            | x86_64, POWER, Arm64            |
  | CUDA cuFFT                                                   | 10.5.1.100          | x86_64, POWER, Arm64            |
  | CUDA cuFile                                                  | 1.0.1.3             | x86_64                          |
  | CUDA cuRAND                                                  | 10.2.5.100          | x86_64, POWER, Arm64            |
  | CUDA cuSOLVER                                                | 11.2.0.100          | x86_64, POWER, Arm64            |
  | CUDA cuSPARSE                                                | 11.6.0.100          | x86_64, POWER, Arm64            |
  | CUDA NPP                                                     | 11.4.0.90           | x86_64, POWER, Arm64            |
  | CUDA nvJPEG                                                  | 11.5.2.100          | x86_64, POWER, Arm64            |
  | Nsight Compute                                               | 2021.2.1.2          | x86_64, POWER, Arm64 (CLI only) |
  | NVTX                                                         | 1.21018621          | x86_64, POWER, Arm64            |
  | Nsight Systems                                               | 2021.2.4.12         | x86_64, POWER, Arm64 (CLI only) |
  | Nsight Visual Studio Edition (VSE)                           | 2021.2.1.21205      | x86_64 (Windows)                |
  | nvidia_fs[1](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#fntarg_1) | 2.7.50              | x86_64                          |
  | Visual Studio Integration                                    | 11.4.100            | x86_64 (Windows)                |
  | NVIDIA Linux Driver                                          | 470.57.02           | x86_64, POWER, Arm64            |
  | NVIDIA Windows Driver                                        | 471.41              | x86_64 (Windows)                |

  CUDA Driver

  Running a CUDA application requires the system with at least one CUDA capable GPU and a driver that is compatible with the CUDA Toolkit. See [Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions). For more information various GPU products that are CUDA capable, visit https://developer.nvidia.com/cuda-gpus.

  Each release of the CUDA Toolkit requires a minimum version of the CUDA driver. The CUDA driver is backward compatible, meaning that applications compiled against a particular version of the CUDA will continue to work on subsequent (later) driver releases.

  More information on compatibility can be found at https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-and-upgrades.

  **Note**: Starting with CUDA 11.0, the toolkit components are individually versioned, and the toolkit itself is versioned as shown in the table below.

  The minimum required driver version for CUDA enhanced compatibility is shown below. CUDA Enhanced Compatibility is described in detail in https://docs.nvidia.com/deploy/cuda-compatibility/index.html

  | CUDA Toolkit                | Minimum Required Driver Version for CUDA Enhanced Compatibility |          |
  | --------------------------- | ------------------------------------------------------------ | -------- |
  | Linux x86_64 Driver Version | Windows x86_64 Driver Version                                |          |
  | CUDA 11.4                   | >=450.80.02                                                  | >=456.38 |
  | CUDA 11.3                   | >=450.80.02                                                  | >=456.38 |
  | CUDA 11.2                   | >=450.80.02                                                  | >=456.38 |
  | CUDA 11.1 (11.1.0)          | >=450.80.02                                                  | >=456.38 |
  | CUDA 11.0 (11.0.3)          | >=450.36.06*                                                 | >=456.38 |

  \* *CUDA 11.0 was released with an earlier driver version, but by upgrading to 450.80.02 driver as indicated, minor version compatibility is possible across the CUDA 11.x family of toolkits.*

  The version of the development NVIDIA GPU Driver packaged in each CUDA Toolkit release is shown below.

  

  | CUDA Toolkit                                      | Toolkit Driver Version        |           |
  | ------------------------------------------------- | ----------------------------- | --------- |
  | Linux x86_64 Driver Version                       | Windows x86_64 Driver Version |           |
  | CUDA 11.4 Update 1                                | >=470.57.02                   | >=471.41  |
  | CUDA 11.4.0 GA                                    | >=470.42.01                   | >=471.11  |
  | CUDA 11.3.1 Update 1                              | >=465.19.01                   | >=465.89  |
  | CUDA 11.3.0 GA                                    | >=465.19.01                   | >=465.89  |
  | CUDA 11.2.2 Update 2                              | >=460.32.03                   | >=461.33  |
  | CUDA 11.2.1 Update 1                              | >=460.32.03                   | >=461.09  |
  | CUDA 11.2.0 GA                                    | >=460.27.03                   | >=460.82  |
  | CUDA 11.1.1 Update 1                              | >=455.32                      | >=456.81  |
  | CUDA 11.1 GA                                      | >=455.23                      | >=456.38  |
  | CUDA 11.0.3 Update 1                              | >= 450.51.06                  | >= 451.82 |
  | CUDA 11.0.2 GA                                    | >= 450.51.05                  | >= 451.48 |
  | CUDA 11.0.1 RC                                    | >= 450.36.06                  | >= 451.22 |
  | CUDA 10.2.89                                      | >= 440.33                     | >= 441.22 |
  | CUDA 10.1 (10.1.105 general release, and updates) | >= 418.39                     | >= 418.96 |
  | CUDA 10.0.130                                     | >= 410.48                     | >= 411.31 |
  | CUDA 9.2 (9.2.148 Update 1)                       | >= 396.37                     | >= 398.26 |
  | CUDA 9.2 (9.2.88)                                 | >= 396.26                     | >= 397.44 |
  | CUDA 9.1 (9.1.85)                                 | >= 390.46                     | >= 391.29 |
  | CUDA 9.0 (9.0.76)                                 | >= 384.81                     | >= 385.54 |
  | CUDA 8.0 (8.0.61 GA2)                             | >= 375.26                     | >= 376.51 |
  | CUDA 8.0 (8.0.44)                                 | >= 367.48                     | >= 369.30 |
  | CUDA 7.5 (7.5.16)                                 | >= 352.31                     | >= 353.66 |
  | CUDA 7.0 (7.0.28)                                 | >= 346.46                     | >= 347.62 |

  

We recommend Anaconda as Python package management system. Please refer to pytorch.org for the detail of PyTorch (torch) installation. The following is the corresponding torchvision versions and supported Python versions.

torch	torchvision	python
main / nightly	main / nightly	>=3.6, <=3.9
1.10.0	0.11.1	>=3.6, <=3.9
1.9.1	0.10.1	>=3.6, <=3.9
1.9.0	0.10.0	>=3.6, <=3.9
1.8.2	0.9.2	>=3.6, <=3.9
1.8.1	0.9.1	>=3.6, <=3.9
1.8.0	0.9.0	>=3.6, <=3.9
1.7.1	0.8.2	>=3.6, <=3.9
1.7.0	0.8.1	>=3.6, <=3.8
1.7.0	0.8.0	>=3.6, <=3.8
1.6.0	0.7.0	>=3.6, <=3.8
1.5.1	0.6.1	>=3.5, <=3.8
1.5.0	0.6.0	>=3.5, <=3.8
1.4.0	0.5.0	==2.7, >=3.5, <=3.8
1.3.1	0.4.2	==2.7, >=3.5, <=3.7
1.3.0	0.4.1	==2.7, >=3.5, <=3.7
1.2.0	0.4.0	==2.7, >=3.5, <=3.7
1.1.0	0.3.0	==2.7, >=3.5, <=3.7
<=1.0.1	0.2.2	==2.7, >=3.5, <=3.7
Anaconda:

conda install torchvision -c pytorch
pip:

pip install torchvision
From source:

python setup.py install
# or, for OSX
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
In case building TorchVision from source fails, install the nightly version of PyTorch following the linked guide on the contributing page and retry the install.

By default, GPU support is built if CUDA is found and torch.cuda.is_available() is true. It's possible to force building GPU support by setting FORCE_CUDA=1 environment variable, which is useful when building a docker image.





