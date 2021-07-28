# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_model_convert SRC DST OUT)
    set(onnx_gen_script ${OpenVINO_SOURCE_DIR}/ngraph/test/models/onnx/onnx_prototxt_converter.py)

    file(GLOB_RECURSE prototxt_models RELATIVE "${SRC}" "${SRC}/*.prototxt")
    file(GLOB_RECURSE xml_models RELATIVE "${SRC}" "${SRC}/*.xml")
    file(GLOB_RECURSE bin_models RELATIVE "${SRC}" "${SRC}/*.bin")
    file(GLOB_RECURSE onnx_models RELATIVE "${SRC}" "${SRC}/*.onnx")

    set(model_files ${prototxt_models} ${xml_models} ${bin_models} ${onnx_models})

    # TODO: these models failed to be converted
    list(REMOVE_ITEM model_files
        # ngraph models
        "models/onnx/external_data/external_data.prototxt"
        "models/onnx/external_data/external_data_different_paths.prototxt"
        "models/onnx/external_data/external_data_file_not_found.prototxt"
        "models/onnx/external_data/external_data_optional_fields.prototxt"
        "models/onnx/external_data/external_data_sanitize_test.prototxt"
        "models/onnx/external_data/external_data_two_tensors_data_in_the_same_file.prototxt"
        "models/onnx/external_data/inner_scope/external_data_file_in_up_dir.prototxt"
        "models/onnx/filename.prototxt"

        # IE models
        "models/onnx_external_data.prototxt"
        "models/support_test/unsupported/no_valid_keys.prototxt"
        )

    foreach(in_file IN LISTS model_files)
        get_filename_component(ext "${in_file}" EXT)
        get_filename_component(rel_dir "${in_file}" DIRECTORY)
        get_filename_component(name_we "${in_file}" NAME_WE)
        set(model_source_dir "${SRC}/${rel_dir}")

        if(NOT NGRAPH_ONNX_IMPORT_ENABLE AND ext MATCHES "^\\.(onnx|prototxt)$")
            # don't copy / process ONNX / prototxt files
            continue()
        endif()

        if(ext STREQUAL ".prototxt")
            # convert model
            set(rel_out_name "${rel_dir}/${name_we}.onnx")
        else()
            # copy as is
            set(rel_out_name "${in_file}")
        endif()

        set(full_out_name "${DST}/${rel_out_name}")
        file(MAKE_DIRECTORY "${DST}/${rel_dir}")

        if(ext MATCHES "^\\.(onnx|bin|xml)$")
            add_custom_command(OUTPUT ${full_out_name}
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SRC}/${in_file}" ${full_out_name}
                DEPENDS ${onnx_gen_script} "${SRC}/${in_file}"
                COMMENT "Copy ${rel_out_name}"
                WORKING_DIRECTORY "${model_source_dir}")
        else()
            # convert .prototxt models to .onnx binary
            add_custom_command(OUTPUT ${full_out_name}
                COMMAND ${PYTHON_EXECUTABLE} ${onnx_gen_script}
                    "${SRC}/${in_file}" ${full_out_name}
                DEPENDS ${onnx_gen_script} "${SRC}/${in_file}"
                COMMENT "Generate ${rel_out_name}"
                WORKING_DIRECTORY "${model_source_dir}")
        endif()
        list(APPEND files "${full_out_name}")
    endforeach()

    set(${OUT} ${files} PARENT_SCOPE)
endfunction()

ov_model_convert("${CMAKE_CURRENT_SOURCE_DIR}/ngraph/test"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/ngraph"
                  onnx_out_files)

set(rel_path "inference-engine/tests/functional/inference_engine/onnx_reader")
ov_model_convert("${OpenVINO_SOURCE_DIR}/${rel_path}"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/onnx_reader"
                 ie_onnx_out_files)

set(rel_path "inference-engine/tests/functional/inference_engine/ir_serialization")
ov_model_convert("${OpenVINO_SOURCE_DIR}/${rel_path}"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/ir_serialization"
                 ie_serialize_out_files)

set(rel_path "inference-engine/tests/unit/frontends/onnx_import/models")
ov_model_convert("${OpenVINO_SOURCE_DIR}/${rel_path}"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/onnx_import"
                 ie_onnx_import_out_files)

add_custom_target(test_model_zoo DEPENDS ${onnx_out_files}
                                         ${ie_onnx_out_files}
                                         ${ie_serialize_out_files}
                                         ${ie_onnx_import_out_files})

install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo"
        DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

set(TEST_MODEL_ZOO "./test_model_zoo" CACHE PATH "Path to test model zoo")
