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

function(ie_cpu_forced_include target graph_header compression_mode)
    if(MSVC)
        set(forced_include "/FI ${graph_header}")
    else()
        set(forced_include "-imacros ${graph_header}")
        if(compression_mode)
            set(forced_include "${forced_include} -Wno-undef")
        endif()
    endif()

    set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${forced_include})
endfunction()

macro(ie_cpu_set_output_directory)
    set_target_properties(${ARGV} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endmacro()

function(ie_cpu_plugin_build)
    set(options ALLOW_UNDEF)
    set(oneValueArgs TARGET_NAME GRAPH_HEADER DEVICE_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(IE_CPU "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # create mkldnn target
    set(SDL_cmake_included ON)
    if(IE_CPU_TARGET_NAME MATCHES ".*_reference$")
        set(TARGET "kernels_reference")
    else()
        set(TARGET "mkldnn")
    endif()
    include(${IE_MAIN_SOURCE_DIR}/thirdparty/mkldnn.cmake)

    # create plugin
    ie_add_plugin(NAME ${IE_CPU_TARGET_NAME}
                  DEVICE_NAME ${IE_CPU_DEVICE_NAME}
                  OBJECT_LIBRARIES ${object_libraries}
                  SOURCES ${SOURCES} ${HEADERS})

    # Cross compiled function
    # TODO: The same for proposal, proposalONNX, topk
    cross_compiled_file(${IE_CPU_TARGET_NAME}
            ARCH AVX512F AVX2 SSE42 ANY
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

    if(NOT "${TARGET}" STREQUAL "mkldnn")
        ie_cpu_set_output_directory(${TARGET} ${IE_CPU_TARGET_NAME})
    endif()

    ie_cpu_forced_include(${TARGET} ${IE_CPU_GRAPH_HEADER} ${IE_CPU_ALLOW_UNDEF})
    ie_cpu_forced_include(${IE_CPU_TARGET_NAME} ${IE_CPU_GRAPH_HEADER} ${IE_CPU_ALLOW_UNDEF})

    set_ie_threading_interface_for(${IE_CPU_TARGET_NAME})

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
    ie_cpu_plugin_build(TARGET_NAME "cpu_reference"
                        DEVICE_NAME "cpu_reference"
                        GRAPH_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp")

    # build tool to generate header file used in custom CPU plugin compilation
    add_executable(ie_cpu_gen_header_tool gen_header.cpp)
    target_link_libraries(ie_cpu_gen_header_tool PRIVATE pugixml
        inference_engine_plugin_api inference_engine ${NGRAPH_LIBRARIES})
    ie_cpu_set_output_directory(ie_cpu_gen_header_tool)

    # build `cpu_reference` target for the ie_cpu_gen_header_tool
    add_dependencies(ie_cpu_gen_header_tool "cpu_reference")

    # run the ie_cpu_gen_header_tool tool
    set(gen_header_file "${CMAKE_CURRENT_BINARY_DIR}/ie_cpu_gen_header.hpp")
    file(REMOVE ${gen_header_file})

    set(gen_header_tool_artifacts "${gen_header_file}")
    foreach(graph IN LISTS CPU_GRAPHS)
        get_filename_component(graph_file_name "${graph}" NAME_WE)
        set(exec_graph_file_ref "${CMAKE_CURRENT_BINARY_DIR}/exec_${graph_file_name}_ref.xml")
        list(APPEND gen_header_tool_artifacts "${exec_graph_file_ref}")
    endforeach()

    add_custom_command(OUTPUT ${gen_header_tool_artifacts}
                       COMMAND $<TARGET_FILE:ie_cpu_gen_header_tool>
                               $<TARGET_FILE:cpu_reference> ${gen_header_file} ${CPU_GRAPHS}
                       DEPENDS cpu_reference ie_cpu_gen_header_tool
                       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                       COMMENT "Generate header file for custom CPU runtime compilation"
                       VERBATIM)
    set_source_files_properties(${gen_header_file} PROPERTIES GENERATED ON)
    add_custom_target(ie_cpu_gen_header DEPENDS ${gen_header_file})

    # build optimized plugin
    ie_cpu_plugin_build(TARGET_NAME "MKLDNNPlugin"
                        DEVICE_NAME "CPU"
                        GRAPH_HEADER ${gen_header_file}
                        ALLOW_UNDEF)
    add_dependencies("MKLDNNPlugin" ie_cpu_gen_header)
    add_dependencies("mkldnn" ie_cpu_gen_header)

    # check that with CPU (not cpu_reference) the executable graphs are generated the same
    set(executable_graphs_actual "")
    foreach(graph IN LISTS CPU_GRAPHS)
        get_filename_component(graph_file_name "${graph}" NAME_WE)
        set(exec_graph_file_actual "${CMAKE_CURRENT_BINARY_DIR}/exec_${graph_file_name}_actual.xml")
        set(exec_graph_file_ref "${CMAKE_CURRENT_BINARY_DIR}/exec_${graph_file_name}_ref.xml")

        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND $<TARGET_FILE:ie_cpu_gen_header_tool> ${graph}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Generate actual executable graph for ${graph_file_name} model"
            VERBATIM)

        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
                    -D "ACTUAL_GRAPH=${exec_graph_file_actual}"
                    -D "REF_GRAPH=${exec_graph_file_ref}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/exec_graph_check.cmake"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Check that execution graph for ${graph_file_name} is valid"
            VERBATIM)
    endforeach()

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
                -D "ACTUAL_LIBRARY=$<TARGET_FILE:MKLDNNPlugin>"
                -D "REF_LIBRARY=$<TARGET_FILE:cpu_reference>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/compression_info.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Compression status ..."
        VERBATIM)
else()
    ie_cpu_plugin_build(TARGET_NAME "MKLDNNPlugin"
                        DEVICE_NAME "CPU"
                        GRAPH_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp")
endif()

#  add test object library

add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})

ie_cpu_forced_include(${TARGET_NAME}_obj "${CMAKE_CURRENT_BINARY_DIR}/ie_gen_default.hpp" OFF)

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:mkldnn,COMPILE_DEFINITIONS>
                                                      USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

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
