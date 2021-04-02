//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include <fstream>

#include "place.hpp"

namespace ngraph {
namespace frontend {

class FrontEndPDPD;

class NGRAPH_API InputModelPDPD : public InputModel
{
    // TODO: replace it by already deserialized proto hidden under some Impl class
    // TODO: avoid using explicit format-dependent data stuctures here, hide it under some Impl class
   
    friend class FrontEndPDPD;

public:
    std::string path;
    std::string model_file;
    std::ifstream weights_stream;
    bool weights_composed = false;

    InputModelPDPD (const std::string& _path) : path(_path) {}
};

} // namespace frontend
} // namespace ngraph
