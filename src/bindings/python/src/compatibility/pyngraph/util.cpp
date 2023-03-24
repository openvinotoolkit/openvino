// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/util.hpp"

#include <pybind11/numpy.h>

#include <transformations/utils/utils.hpp>

#include "ngraph/op/result.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/version.hpp"
#include "openvino/runtime/core.hpp"

namespace py = pybind11;

inline void* numpy_to_c(py::array a) {
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
        return ov::op::util::get_ie_output_name(output);
    });

    mod.def("shutdown",
            &ov::shutdown,
            R"(
                    Shut down the OpenVINO by deleting all static-duration objects allocated by the library and releasing
                    dependent resources

                    This function should be used by advanced user to control unload the resources.

                    You might want to use this function if you are developing a dynamically-loaded library which should clean up all
                    resources after itself when the library is unloaded.
                )");
}
