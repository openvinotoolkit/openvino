# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if (NOT NGRAPH_PYTHON_BUILD_ENABLE)
    message(WARNING "Please enable nGraph Python API (_pyngraph) target to enable Model Optimizer target")
elseif(NOT ENABLE_PYTHON)
    message(WARNING "Please enable IE Python API (ie_api and offline_transformations_api) targets to enable Model Optimizer target")
else()
    add_custom_target(model_optimizer DEPENDS ie_api offline_transformations_api inference_engine_ir_reader)
    if(ENABLE_TESTS)
            add_subdirectory(unit_tests/mock_mo_frontend/mock_mo_ngraph_frontend)
            add_dependencies(model_optimizer mock_mo_ngraph_frontend)

            add_subdirectory(unit_tests/mock_mo_frontend/mock_mo_python_api)
            add_dependencies(model_optimizer mock_mo_python_api)
    endif()
endif()

# install
ie_cpack_add_component(model_optimizer)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/automation/version.txt.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.txt"
    @ONLY)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION deployment_tools/model_optimizer
        USE_SOURCE_PERMISSIONS
        COMPONENT model_optimizer
        PATTERN ".*" EXCLUDE
        PATTERN "automation" EXCLUDE
        PATTERN "requirements_dev.txt" EXCLUDE
        PATTERN "README.md" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        
        PATTERN "extensions/front/caffe/CustomLayersMapping.xml" EXCLUDE
        PATTERN "mo/utils/convert.py" EXCLUDE
        PATTERN "unit_tests" EXCLUDE
        
        REGEX ".*__pycache__.*" EXCLUDE
        REGEX ".*\\.pyc$" EXCLUDE
        REGEX ".*\\.swp" EXCLUDE
        REGEX ".*\\.DS_Store$" EXCLUDE
        REGEX ".*_test\.py$" EXCLUDE
        )
