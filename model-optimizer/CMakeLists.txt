# Copyright (C) 2018-2020 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# install
ie_cpack_add_component(model_optimizer REQUIRED)

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
        PATTERN "mo/utils/unittest" EXCLUDE
        
        REGEX ".*__pycache__.*" EXCLUDE
        REGEX ".*\\.pyc$" EXCLUDE
        REGEX ".*\\.swp" EXCLUDE
        REGEX ".*\\.DS_Store$" EXCLUDE
        REGEX ".*_test\.py$" EXCLUDE
        )
