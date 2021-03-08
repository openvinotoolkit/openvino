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
