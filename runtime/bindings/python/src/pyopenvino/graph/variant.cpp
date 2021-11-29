// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/variant.hpp"  // ov::Variant

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/variant.hpp"

namespace py = pybind11;

void regclass_graph_Variant(py::module m) {
    py::class_<ov::Variant, std::shared_ptr<ov::Variant>> variant_base(m, "Variant");
    variant_base.doc() = "openvino.impl.Variant wraps ov::Variant";
}

template void regclass_graph_VariantWrapper<std::string>(py::module m, std::string typestring);
template void regclass_graph_VariantWrapper<int64_t>(py::module m, std::string typestring);
