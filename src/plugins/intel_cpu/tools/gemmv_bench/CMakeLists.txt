cmake_minimum_required(VERSION 3.16)

project(gemmv_bench CXX)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

add_executable(gemmv_bench
    main.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/entry.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/gemmv_ukernel.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/jit_gemmv_avx512_fp32.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/jit_gemmv_avx512_vnni_s32.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/vnni_gemmv_intrin.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/jit_minigemm_avx512_fp32.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/jit_gemmv_avx512_simple.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/jit_gemmv_avx2_fp32.cpp
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/amx_gemmv_intrin.cpp
)

# Enable AMX intrinsics for this translation unit (if compiler supports)
set_source_files_properties(
    ${SRC_DIR}/nodes/kernels/x64/gemmv_jit/amx_gemmv_intrin.cpp
    PROPERTIES COMPILE_OPTIONS "-mamx-int8;-mamx-bf16;-mamx-tile;-mavx512f"
)

target_include_directories(gemmv_bench PRIVATE ${SRC_DIR})

# Provide a fallback interface target for oneDNN includes when building this subproject directly
if(NOT TARGET dnnl)
    add_library(dnnl INTERFACE)
    target_include_directories(dnnl INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onednn/include)
endif()
target_include_directories(gemmv_bench SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onednn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onednn/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..//thirdparty/xbyak)

target_compile_features(gemmv_bench PRIVATE cxx_std_17)

target_link_libraries(gemmv_bench PRIVATE dnnl)

set_target_properties(gemmv_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Convenience target: run comparison and generate CSV
add_custom_target(compare
    COMMAND ${CMAKE_COMMAND} -E env GEMMV_BIN=$<TARGET_FILE:gemmv_bench> bash ${CMAKE_CURRENT_SOURCE_DIR}/gemmv_vs_benchdnn.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running gemmv_vs_benchdnn.sh to generate comparison CSV"
    VERBATIM)
