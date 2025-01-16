// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/layout_helpers.hpp"

#include <pybind11/pybind11.h>

#include "openvino/core/layout.hpp"

namespace py = pybind11;

void regmodule_graph_layout_helpers(py::module m) {
    py::module mod = m.def_submodule("layout_helpers", "openvino.layout_helpers");

    mod.def("has_batch", &ov::layout::has_batch, py::arg("layout"));
    mod.def("batch_idx", &ov::layout::batch_idx, py::arg("layout"));
    mod.def("has_channels", &ov::layout::has_channels, py::arg("layout"));
    mod.def("channels_idx", &ov::layout::channels_idx, py::arg("layout"));
    mod.def("has_depth", &ov::layout::has_depth, py::arg("layout"));
    mod.def("depth_idx", &ov::layout::depth_idx, py::arg("layout"));
    mod.def("has_height", &ov::layout::has_height, py::arg("layout"));
    mod.def("height_idx", &ov::layout::height_idx, py::arg("layout"));
    mod.def("has_width", &ov::layout::has_width, py::arg("layout"));
    mod.def("width_idx", &ov::layout::width_idx, py::arg("layout"));
    mod.def("get_layout",
            static_cast<ov::Layout (*)(const ov::Output<ov::Node>&)>(&ov::layout::get_layout),
            py::arg("port"));
    mod.def("get_layout",
            static_cast<ov::Layout (*)(const ov::Output<const ov::Node>&)>(&ov::layout::get_layout),
            py::arg("port"));
    mod.def("set_layout", &ov::layout::set_layout, py::arg("port"), py::arg("layout"));
}
