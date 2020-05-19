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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "pyngraph/runtime/tensor.hpp"

namespace py = pybind11;

static void read_(ngraph::runtime::Tensor* self, void* p, size_t n)
{
    self->read(p, n);
}

static void write_(ngraph::runtime::Tensor* self, void* p, size_t n)
{
    self->write(p, n);
}

void regclass_pyngraph_runtime_Tensor(py::module m)
{
    py::class_<ngraph::runtime::Tensor, std::shared_ptr<ngraph::runtime::Tensor>> tensor(m,
                                                                                         "Tensor");
    tensor.doc() = "ngraph.impl.runtime.Tensor wraps ngraph::runtime::Tensor";
    tensor.def("write", &write_);
    tensor.def("read", &read_);

    tensor.def_property_readonly("shape", &ngraph::runtime::Tensor::get_shape);
    tensor.def_property_readonly("element_count", &ngraph::runtime::Tensor::get_element_count);
    tensor.def_property_readonly("element_type", [](const ngraph::runtime::Tensor& self) {
        return self.get_element_type();
    });
}
