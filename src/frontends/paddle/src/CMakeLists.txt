# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Find required packages
find_package(nlohmann_json REQUIRED)
find_package(yaml-cpp REQUIRED)

ov_add_frontend(NAME paddle
                LINKABLE_FRONTEND
                PROTOBUF_REQUIRED
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format"
                SOURCES
                    decoder.cpp
                    decoder_json.cpp
                    decoder_json.hpp
                    json_input_model_imp.cpp
                    json_input_model_imp.hpp
                    yaml_metadata.cpp
                    yaml_metadata.hpp
                LINK_LIBRARIES 
                    openvino::util 
                    openvino::core::dev
                    nlohmann_json::nlohmann_json
                    yaml-cpp::yaml-cpp)

ov_build_target_faster(openvino_paddle_frontend PCH)
