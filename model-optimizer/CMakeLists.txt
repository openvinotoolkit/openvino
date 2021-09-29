# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(NOT ENABLE_PYTHON)
    message(WARNING "Please enable IE & nGraph Python API (ie_api and offline_transformations_api) targets to enable Model Optimizer target")
else()
    add_custom_target(model_optimizer DEPENDS ie_api offline_transformations_api ir_ngraph_frontend)
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
        DESTINATION tools/model_optimizer
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
        PATTERN "openvino_mo.egg-info" EXCLUDE
        PATTERN "build" EXCLUDE

        REGEX ".*__pycache__.*" EXCLUDE
        REGEX ".*\\.pyc$" EXCLUDE
        REGEX ".*\\.swp" EXCLUDE
        REGEX ".*\\.DS_Store$" EXCLUDE
        REGEX ".*_test\.py$" EXCLUDE
        )

install(FILES requirements_dev.txt
        DESTINATION tools/model_optimizer
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(DIRECTORY unit_tests
        DESTINATION tools/model_optimizer
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(DIRECTORY automation
        DESTINATION tools/model_optimizer
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/.coveragerc
        DESTINATION tools/model_optimizer
        COMPONENT tests
        EXCLUDE_FROM_ALL)
