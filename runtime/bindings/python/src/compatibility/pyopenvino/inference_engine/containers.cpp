
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pyopenvino/core/containers.hpp"

PYBIND11_MAKE_OPAQUE(Containers::PyConstInputsDataMap);
PYBIND11_MAKE_OPAQUE(Containers::PyOutputsDataMap);
PYBIND11_MAKE_OPAQUE(Containers::PyResults);

namespace py = pybind11;

namespace Containers {

    void regclass_PyConstInputsDataMap(py::module m) {
        auto py_const_inputs_data_map = py::bind_map<PyConstInputsDataMap>(m, "PyConstInputsDataMap");

        py_const_inputs_data_map.def("keys", [](PyConstInputsDataMap& self) {
            return py::make_key_iterator(self.begin(), self.end());
        });
    }

    void regclass_PyOutputsDataMap(py::module m) {
        auto py_outputs_data_map = py::bind_map<PyOutputsDataMap>(m, "PyOutputsDataMap");

        py_outputs_data_map.def("keys", [](PyOutputsDataMap& self) {
            return py::make_key_iterator(self.begin(), self.end());
        });
    }

    void regclass_PyResults(py::module m) {
        auto py_results = py::bind_map<PyResults>(m, "PyResults");

        py_results.def("keys", [](PyResults& self) {
            return py::make_key_iterator(self.begin(), self.end());
        });
    }
}
