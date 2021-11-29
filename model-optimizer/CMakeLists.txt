# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
