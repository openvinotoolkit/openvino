// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/util.hpp"

#include <pybind11/numpy.h>

#include <transformations/utils/utils.hpp>

#include "ngraph/op/result.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/version.hpp"

namespace py = pybind11;

void* numpy_to_c(py::array a) {
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_pyngraph_util(py::module m) {
    py::module mod = m.def_submodule("util", "ngraph.impl.util");
    mod.def("numpy_to_c", &numpy_to_c);
    mod.def("get_constant_from_source",
            &ngraph::get_constant_from_source,
            py::arg("output"),
            R"(
                    Runs an estimation of source tensor.
                    Parameters
                    ----------
                    output : Output
                        output node
                    Returns
                    ----------
                    get_constant_from_source : Constant or None
                        If it succeeded to calculate both bounds and
                        they are the same returns Constant operation
                        from the resulting bound, otherwise Null.
                )");

    mod.def("get_ngraph_version_string", []() -> std::string {
        NGRAPH_SUPPRESS_DEPRECATED_START
        return get_ngraph_version_string();
        NGRAPH_SUPPRESS_DEPRECATED_END
    });

    mod.def("get_ie_output_name", [](const ngraph::Output<ngraph::Node>& output) {
        return ngraph::op::util::get_ie_output_name(output);
    });
}
