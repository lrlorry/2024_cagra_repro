cmake_minimum_required(VERSION 3.26)
project(cuvs_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU 架构：RTX 5090 = sm_100（Blackwell），也兼容 Ampere/Ada。
# 用 -DCMAKE_CUDA_ARCHITECTURES=100 覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100")
endif()

# cuVS 通过 conda 安装后可通过 find_package 找到。
# 构建时传 -DCMAKE_PREFIX_PATH=$CONDA_PREFIX 即可。
find_package(cuvs REQUIRED)

add_executable(cuvs_bench cuvs_bench.cu)
target_link_libraries(cuvs_bench PRIVATE cuvs::cuvs)

# 和 engineered_core 保持一致的编译选项
target_compile_options(cuvs_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 --generate-line-info>
)
