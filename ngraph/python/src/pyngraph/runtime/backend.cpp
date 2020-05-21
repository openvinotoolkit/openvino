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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "pyngraph/runtime/backend.hpp"

namespace py = pybind11;

static std::shared_ptr<ngraph::runtime::Backend> create_static(const std::string& type)
{
    bool must_support_dynamic = false;
    return ngraph::runtime::Backend::create(type, must_support_dynamic);
}

static std::shared_ptr<ngraph::runtime::Backend> create_dynamic(const std::string& type)
{
    bool must_support_dynamic = true;
    return ngraph::runtime::Backend::create(type, must_support_dynamic);
}

static std::shared_ptr<ngraph::runtime::Executable> compile(ngraph::runtime::Backend* self,
                                                            std::shared_ptr<ngraph::Function> func)
{
    bool enable_performance_data = false;
    return self->compile(func, enable_performance_data);
}

void regclass_pyngraph_runtime_Backend(py::module m)
{
    py::class_<ngraph::runtime::Backend, std::shared_ptr<ngraph::runtime::Backend>> backend(
        m, "Backend");
    backend.doc() = "ngraph.impl.runtime.Backend wraps ngraph::runtime::Backend";
    backend.def_static("create", &create_static);
    backend.def_static("create_dynamic", &create_dynamic);
    backend.def_static("get_registered_devices", &ngraph::runtime::Backend::get_registered_devices);
    backend.def("create_tensor",
                (std::shared_ptr<ngraph::runtime::Tensor>(ngraph::runtime::Backend::*)(
                    const ngraph::element::Type&, const ngraph::Shape&)) &
                    ngraph::runtime::Backend::create_tensor);
    backend.def("create_dynamic_tensor",
                (std::shared_ptr<ngraph::runtime::Tensor>(ngraph::runtime::Backend::*)(
                    const ngraph::element::Type&, const ngraph::PartialShape&)) &
                    ngraph::runtime::Backend::create_dynamic_tensor);
    backend.def("compile", &compile);
    backend.def("set_config", &ngraph::runtime::Backend::set_config);
}
