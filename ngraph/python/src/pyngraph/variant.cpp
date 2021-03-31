// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include "ngraph/variant.hpp" // ngraph::Variant
#include "pyngraph/variant.hpp"

namespace py = pybind11;

void regclass_pyngraph_Variant(py::module m)
{
    py::class_<ngraph::Variant, std::shared_ptr<ngraph::Variant>> variant_base(m, "Variant");
    variant_base.doc() = "ngraph.impl.Variant wraps ngraph::Variant";
}

template void regclass_pyngraph_VariantWrapper<std::string>(py::module m, std::string typestring);
template void regclass_pyngraph_VariantWrapper<int64_t>(py::module m, std::string typestring);
