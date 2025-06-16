# 使用CMake编译
cmake_minimum_required(VERSION 3.5...3.27)
project(contnet_deploy)

# set(CMAKE_PREFIX_PATH "/home/chenjiayi/code/params-tuning/deploy/libtorch")
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
add_executable(inference inference.cpp)
target_link_libraries(inference ${TORCH_LIBRARIES})
target_link_libraries(inference ${YAML_CPP_LIBRARIES})
target_link_libraries(inference ${NLOHMANN_JSON})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)