# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

set(LAYERS
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_activation_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_batchnorm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_bin_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_concat_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_crop_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_deconv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_def_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_depthwise_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_eltwise_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_fullyconnected_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gemm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_generic_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_input_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_lrn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_memory_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_permute_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_power_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_quantize_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reorder_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reshape_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roi_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_softmax_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_split_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tensoriterator_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tile_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_mvn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_resample_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_normalize_node.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_greedy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/extract_image_patches.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/grn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/log_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/one_hot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/powerfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox_clustered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorgridgenerator_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/psroi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/region_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reverse_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/roifeatureextractor_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/simplernms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/space_to_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/space_to_depth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_fill_empty_rows.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_segment_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_weighted_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_to_dense.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/bucketize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topkrois_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unique.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unsqueeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/interp.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/argmax_imp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_imp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/cum_sum.cpp
)

foreach(LAYER ${LAYERS})
    get_filename_component(LAYER_NAME ${LAYER} NAME_WE)
    string(TOUPPER ${LAYER_NAME} LAYER_UNAME)
    add_definitions(-DCOMPILED_CPU_${LAYER_UNAME})
endforeach()

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${LAYERS}
        ${OS_SPECIFIC_SRC}
)

list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gen_header.cpp)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.hpp
)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER MKL_VERSION)

include_directories(
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include)

include_directories(SYSTEM
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include)

if (GEMM STREQUAL "MKL")
    log_rpath_from_dir(MKL "${MKL}/lib")
endif()

if (THREADING STREQUAL "TBB")
    set(MKLDNN_THR MKLDNN_THR_TBB)
elseif (THREADING STREQUAL "TBB_AUTO")
    set(MKLDNN_THR MKLDNN_THR_TBB_AUTO)
elseif (THREADING STREQUAL "OMP")
    set(MKLDNN_THR MKLDNN_THR_OMP)
else()
    set(MKLDNN_THR MKLDNN_THR_SEQ)
endif()

function(ie_cpu_plugin_build)
    set(options)
    set(oneValueArgs TARGET_NAME GRAPH_HEADER DEVICE_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(IE_CPU "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # create mkldnn target
    set(SDL_cmake_included ON)
    if(IE_CPU_TARGET_NAME STREQUAL "MKLDNNPlugin_FULL")
        set(TARGET "mkldnn_full")
    else()
        set(TARGET "mkldnn")
    endif()
    include(${IE_MAIN_SOURCE_DIR}/thirdparty/mkldnn.cmake)

    # create plugin
    ie_add_plugin(NAME ${IE_CPU_TARGET_NAME}
                  DEVICE_NAME ${IE_CPU_DEVICE_NAME}
                  OBJECT_LIBRARIES ${object_libraries}
                  SOURCES ${SOURCES} ${HEADERS})

    if(IE_CPU_GRAPH_HEADER)
        if(MSVC)
            set(forced_include "/FI ${IE_CPU_GRAPH_HEADER}")
        else()
            set(forced_include "-imacros ${IE_CPU_GRAPH_HEADER}")
        endif()

        set_target_properties(${TARGET} ${IE_CPU_TARGET_NAME} PROPERTIES COMPILE_FLAGS ${forced_include})
    endif()

    # Cross compiled function
    # TODO: The same for proposal, proposalONNX, topk
    cross_compiled_file(${IE_CPU_TARGET_NAME}
            ARCH AVX2 SSE42 ANY # TODO: return back AVX512F
                        nodes/argmax_imp.cpp
            API         nodes/argmax_imp.hpp
            NAME        arg_max_execute
            NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
    )
    cross_compiled_file(${IE_CPU_TARGET_NAME}
            ARCH AVX2 ANY
                        nodes/proposal_imp.cpp
            API         nodes/proposal_imp.hpp
            NAME        proposal_exec
            NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
    )

    set_ie_threading_interface_for(${IE_CPU_TARGET_NAME})

    target_compile_definitions(${IE_CPU_TARGET_NAME} PUBLIC -DMKLDNN_THR=${MKLDNN_THR})
    target_link_libraries(${IE_CPU_TARGET_NAME} PRIVATE inference_engine inference_engine_lp_transformations
                                                        inference_engine_transformations
                                                        ${INTEL_ITT_LIBS} ${TARGET})
endfunction()

# build

# variable to hold paths to original graphs which want to optimize CPU runtime size for 
set(CPU_GRAPHS "" CACHE STRING "Paths to graphs to optimize CPU runtime size for")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp" "#define GraphGen(...) 1")

if(CPU_GRAPHS)
    # used to generate executable graph
    ie_cpu_plugin_build(TARGET_NAME "MKLDNNPlugin_FULL"
                        DEVICE_NAME "CPU_FULL"
                        GRAPH_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp")

    # build tool to generate header file used in custom CPU plugin compilation
    add_executable(ie_cpu_gen_header gen_header.cpp)
    target_link_libraries(ie_cpu_gen_header PRIVATE pugixml
        inference_engine_plugin_api inference_engine)

    # build CPU_FULL target for the ie_cpu_gen_header
    add_dependencies(ie_cpu_gen_header MKLDNNPlugin_FULL)

    # generate executable graphs
    function(ie_cpu_generate_executable_graph graph exec_graph_file_out)
        get_filename_component(graph_file_name "${graph}" NAME)
        set(exec_graph_file "${CMAKE_CURRENT_BINARY_DIR}/exec_${graph_file_name}")
        add_custom_command(OUTPUT ${exec_graph_file}
            COMMAND $<TARGET_FILE:ie_cpu_gen_header> ${graph} ${exec_graph_file}
            DEPENDS ie_cpu_gen_header
            COMMENT "Generate executable graph for ${graph_file_name} model"
            VERBATIM)
        set("${exec_graph_file_out}" ${exec_graph_file} PARENT_SCOPE)
    endfunction()

    set(executable_graphs "")
    foreach(graph IN LISTS CPU_GRAPHS)
        ie_cpu_generate_executable_graph(${graph} executable_graph)
        list(APPEND executable_graphs ${executable_graph})
    endforeach()

    # create target to generate executable graphs
    add_custom_target(ie_cpu_generate_executable_graphs DEPENDS ${executable_graphs})

    # run the ie_cpu_gen_header tool
    set(gen_header_file "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_header.hpp")
    file(REMOVE ${gen_header_file})
    add_custom_command(OUTPUT ${gen_header_file}
                       COMMAND $<TARGET_FILE:ie_cpu_gen_header> ${executable_graphs} ${gen_header_file}
                       DEPENDS gen_header.cpp ${executable_graphs}
                       COMMENT "Generate header file for custom CPU runtime compilation"
                       VERBATIM)
    set_source_files_properties(${gen_header_file} PROPERTIES GENERATED ON)
    add_custom_target(ie_gen_header DEPENDS ${gen_header_file})

    # build optimized plugin
    ie_cpu_plugin_build(TARGET_NAME "MKLDNNPlugin"
                        DEVICE_NAME "CPU"
                        GRAPH_HEADER ${gen_header_file})
    add_dependencies("MKLDNNPlugin" ie_gen_header)
    add_dependencies("mkldnn" ie_gen_header)

    # check that with CPU (not CPU_FULL) the executable graphs are generated the same
else()
    ie_cpu_plugin_build(TARGET_NAME "MKLDNNPlugin"
                        DEVICE_NAME "CPU"
                        GRAPH_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp")
endif()

#  add test object library

add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PUBLIC -DMKLDNN_THR=${MKLDNN_THR}
                                              PRIVATE USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

set(mkldnn_plugin_object_libraries "${TARGET_NAME}_obj" CACHE INTERNAL "" FORCE)

# install

if(GEMM STREQUAL "MKL")
    install(DIRECTORY "${MKL}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx
            COMPONENT cpu)
    install(FILES "${MKLLIB}"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx/lib
            COMPONENT cpu)
    install(FILES "${MKL}/version.info"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx
            COMPONENT cpu)
endif()
