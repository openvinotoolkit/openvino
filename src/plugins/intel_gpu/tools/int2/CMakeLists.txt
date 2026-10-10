# Standalone build: point OpenVINO_DIR at the OpenVINO build directory.
#   cmake -B build -DOpenVINO_DIR=<ov-build> -DCMAKE_CXX_COMPILER=icpx && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(int2_tools CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenVINO REQUIRED)

# paged_bench_llm      Qwen3-family decoders (Bonsai 1.7B-8B, CAT-Q): greedy generation with TTFT / decode timing
# Bonsai 27B-class decoders (separate embedding model, rank-3 mrope, paged attention):
# paged_bench_llm_27b  single-sequence greedy generation with TTFT / decode timing
# paged_serve_llm_27b  batched (continuous-batching) generation from a request file, for eval harnesses
foreach(tool paged_bench_llm paged_bench_llm_27b paged_serve_llm_27b)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE openvino::runtime)
endforeach()

# paged_serve_llm_27b zeroes one slot of a state table with the plugin's ROI
# copy, which is only reachable through the dev API.
get_filename_component(OV_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
target_include_directories(paged_serve_llm_27b PRIVATE
    "${OV_SRC_ROOT}/src/inference/dev_api" "${OV_SRC_ROOT}/src/core/dev_api")
