// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/opsets/opset.hpp"
#include "pyngraph/opset.hpp"

namespace py = pybind11;

void regclass_pyngraph_OpSet(py::module m) {
    py::class_<ngraph::OpSet, std::shared_ptr<ngraph::OpSet>> opset(m, "OpSet", py::dynamic_attr());
    opset.doc() = "ngraph.impl.OpSet wraps ngraph::OpSet";
    opset.def(py::init<>());
    opset.def("size",
              &ngraph::OpSet::size,
              R"(
                Get number of operations in opset.

                Returns
                ----------
                size : int
                    Opset size.
               )");
    opset.def("insert",
              static_cast<void(ngraph::OpSet *)(const std::string &, const ngraph::NodeTypeInfo &,
                                                ngraph::FactoryRegistry<ngraph::Node>::Factory)>(&ngraph::OpSet::insert)
    R"(
                Get number of operations in opset.

                Returns
                ----------
                size : int
                    Opset size.
               )");
}
