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

#include <ie_layouts.h>
#include <ie_common.h>
#include <ie_precision.hpp>

#include "pyopenvino/inference_engine/tensor_description.hpp"
#include <pybind11/stl.h>
#include "common.hpp"

namespace py = pybind11;
using namespace InferenceEngine;


void regclass_TensorDecription(py::module m)
{
    py::class_<TensorDesc, std::shared_ptr<TensorDesc>> cls(m, "TensorDesc");
    cls.def(py::init<const Precision&, const SizeVector&, Layout>());
    cls.def(py::init([](const std::string& precision, const SizeVector& dims, const std::string& layout) {
        return TensorDesc(Precision::FromStr(precision), dims, Common::get_layout_from_string(layout));
    }));

    cls.def_property("layout", [](TensorDesc& self) {
        return Common::get_layout_from_enum(self.getLayout());
    }, [](TensorDesc& self, const std::string& layout) {
        self.setLayout(Common::get_layout_from_string(layout));
    });

    cls.def_property("precision", [](TensorDesc& self) {
        return self.getPrecision().name();
        }, [](TensorDesc& self, const std::string& precision) {
        self.setPrecision(InferenceEngine::Precision::FromStr(precision));
    });

    cls.def_property("dims", [](TensorDesc& self) {
        return self.getDims();
        }, [](TensorDesc& self,const SizeVector& dims) {
        self.setDims(dims);
    });

    cls.def("__eq__", [](const TensorDesc& a, const TensorDesc b) {
        return a == b;
        }, py::is_operator());
}
