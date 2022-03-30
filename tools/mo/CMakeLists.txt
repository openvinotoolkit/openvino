# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(NOT ENABLE_PYTHON)
    message(WARNING "Please enable IE & nGraph Python API (pyopenvino) targets to enable Model Optimizer target")
else()
    add_custom_target(model_optimizer DEPENDS ie_api openvino_ir_frontend pyopenvino)
    if(ENABLE_TESTS)
        add_subdirectory(unit_tests/mock_mo_frontend/mock_mo_frontend)
        add_dependencies(model_optimizer openvino_mock_mo_frontend)

        add_subdirectory(unit_tests/mock_mo_frontend/mock_mo_python_api)
        add_dependencies(model_optimizer mock_mo_python_api)
    endif()
endif()

# install

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/automation/version.txt.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" @ONLY)

install(FILES requirements_dev.txt
        DESTINATION tests/mo
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(DIRECTORY unit_tests
        DESTINATION tests/mo
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(DIRECTORY automation
        DESTINATION tests/mo
        COMPONENT tests
        EXCLUDE_FROM_ALL)

install(FILES .coveragerc
        DESTINATION tests/mo
        COMPONENT tests
        EXCLUDE_FROM_ALL)
