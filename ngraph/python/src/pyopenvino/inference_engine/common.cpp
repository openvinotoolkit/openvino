//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <unordered_map>

#include "common.hpp"

namespace Common {

    namespace {
        const std::unordered_map<int, std::string> layout_int_to_str_map = {{0, "ANY"}, {1,"NCHW"},
                                                            {2, "NHWC"}, {3, "NCDHW"},
                                                            {4, "NDHWC"}, {64, "OIHW"},
                                                            {95, "SCALAR"}, {96, "C"},
                                                            {128, "CHW"}, {192, "HW"},
                                                            {193, "NC"}, {194, "CN"},
                                                            {200, "BLOCKED"}};

        const std::unordered_map<std::string, InferenceEngine::Layout> layout_str_to_enum = {
                    {"ANY", InferenceEngine::Layout::ANY},
                    {"NHWC", InferenceEngine::Layout::NHWC},
                    {"NCHW", InferenceEngine::Layout::NCHW},
                    {"NCDHW", InferenceEngine::Layout::NCDHW},
                    {"NDHWC", InferenceEngine::Layout::NDHWC},
                    {"OIHW", InferenceEngine::Layout::OIHW},
                    {"GOIHW", InferenceEngine::Layout::GOIHW},
                    {"OIDHW", InferenceEngine::Layout::OIDHW},
                    {"GOIDHW", InferenceEngine::Layout::GOIDHW},
                    {"SCALAR", InferenceEngine::Layout::SCALAR},
                    {"C", InferenceEngine::Layout::C},
                    {"CHW", InferenceEngine::Layout::CHW},
                    {"HW", InferenceEngine::Layout::HW},
                    {"NC", InferenceEngine::Layout::NC},
                    {"CN", InferenceEngine::Layout::CN},
                    {"BLOCKED", InferenceEngine::Layout::BLOCKED}
            };
    }

    InferenceEngine::Layout get_layout_from_string(const std::string& layout) {
        return layout_str_to_enum.at(layout);
    }

    const std::string& get_layout_from_enum(const InferenceEngine::Layout &layout) {
        return layout_int_to_str_map.at(layout);
    }
};
