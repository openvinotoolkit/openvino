// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
