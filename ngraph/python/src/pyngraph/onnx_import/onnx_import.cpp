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

#if defined(NGRAPH_ONNX_IMPORT_ENABLE)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/function.hpp"
#include "pyngraph/onnx_import/onnx_import.hpp"

namespace py = pybind11;

static std::shared_ptr<ngraph::Function> import_onnx_model(const std::string& model_proto)
{
    std::istringstream iss(model_proto, std::ios_base::binary | std::ios_base::in);
    return ngraph::onnx_import::import_onnx_model(iss);
}

static std::shared_ptr<ngraph::Function> import_onnx_model_file(const std::string& filename)
{
    return ngraph::onnx_import::import_onnx_model(filename);
}

void regmodule_pyngraph_onnx_import(py::module mod)
{
    mod.def("import_onnx_model", &import_onnx_model);
    mod.def("import_onnx_model_file", &import_onnx_model_file);
}
#endif
