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

cmake_minimum_required(VERSION 3.10)

file(REMOVE "${CMAKE_BINARY_DIR}/openvino_targets_developer.cmake")

unset(OpenVINODeveloperPackageTargets CACHE)

function(openvino_developer_export_targets)
    set(OpenVINODeveloperPackageTargets "${OpenVINODeveloperPackageTargets};${ARGV}")

    # to allow exporting of aliased targets with the original names
    foreach(target_name ${OpenVINODeveloperPackageTargets})
        if(TARGET "${target_name}")
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET "${original_name}")
                message(STATUS "The name ${target_name} is an ALIAS for ${original_name}. "
                        "It will be exported to the InferenceEngineDeveloperPackage with the original name.")
                list(REMOVE_ITEM OpenVINODeveloperPackageTargets ${target_name})
                list(APPEND OpenVINODeveloperPackageTargets ${original_name})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES OpenVINODeveloperPackageTargets)
    set(OpenVINODeveloperPackageTargets "${OpenVINODeveloperPackageTargets}" CACHE INTERNAL
        "Paths to extra Inference Engine plugins" FORCE)
endfunction()

add_subdirectory(itt)

openvino_developer_export_targets(openvino::itt)
