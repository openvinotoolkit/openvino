// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/preprocess/pre_post_process.hpp"

namespace py = pybind11;

namespace pybind11 { namespace detail {

template <> struct type_caster<ov::Layout> : public type_caster_base<ov::Layout> {
    using base = type_caster_base<ov::Layout>;
public:
    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a LayoutWrapper 
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert) {
        if (base::load(src, convert)) {
            return true;
        }
        else if (py::isinstance<py::str>(src)) {
            value = new ov::Layout(py::cast<std::string>(src));
            return true;
        }

        return false;
    }
    /**
     * Conversion part 2 (C++ -> Python): convert an LayoutWrapper instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(ov::Layout src, return_value_policy policy, handle parent) {
        return pybind11::cast(src, policy, parent);
    }
};
}
}

void regclass_graph_PrePostProcessor(py::module m);
