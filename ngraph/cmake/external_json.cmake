# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download json
#------------------------------------------------------------------------------

SET(JSON_GIT_REPO_URL https://github.com/nlohmann/json)
if(WIN32)
    SET(JSON_GIT_LABEL v3.5.0)
else()
    SET(JSON_GIT_LABEL v3.7.3)
endif()

ExternalProject_Add(
    ext_json
    PREFIX json
    GIT_REPOSITORY ${JSON_GIT_REPO_URL}
    GIT_TAG ${JSON_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_json SOURCE_DIR)
add_library(libjson INTERFACE)
target_include_directories(libjson SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libjson ext_json)
