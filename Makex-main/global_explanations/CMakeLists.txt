cmake_minimum_required(VERSION 3.10)
project(sarl_miner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 Torch 路径（若未安装到默认位置，可通过 -DCMAKE_PREFIX_PATH=... 传入）
find_package(Torch REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

add_executable(sarl_miner sarl_miner.cpp)
target_link_libraries(sarl_miner "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
target_compile_options(sarl_miner PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(sarl_miner PRIVATE ${OpenMP_CXX_FLAGS})

# 避免 RPATH 问题
set_property(TARGET sarl_miner PROPERTY CXX_STANDARD 17)
set_property(TARGET sarl_miner PROPERTY POSITION_INDEPENDENT_CODE ON)
