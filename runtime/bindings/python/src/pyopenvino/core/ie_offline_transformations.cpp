// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/ie_offline_transformations.hpp"

#include <generate_mapping_file.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <pot_transformations.hpp>
#include <pruning.hpp>

#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"

namespace py = pybind11;

void regmodule_offline_transformations(py::module m) {
    py::module m_offline_transformations = m.def_submodule("offline_transformations", "Offline transformations module");

    m_offline_transformations.def(
        "ApplyMOCTransformations",
        [](std::shared_ptr<ov::Function> function, bool cf) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::MOCTransformations>(cf);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("cf"));

    m_offline_transformations.def(
        "ApplyPOTTransformations",
        [](std::shared_ptr<ov::Function> function, std::string device) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::POTTransformations>(std::move(device));
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("device"));

    m_offline_transformations.def(
        "ApplyLowLatencyTransformation",
        [](std::shared_ptr<ov::Function> function, bool use_const_initializer = true) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::LowLatency2>(use_const_initializer);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("use_const_initializer") = true);

    m_offline_transformations.def(
        "ApplyPruningTransformation",
        [](std::shared_ptr<ngraph::Function> function) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::Pruning>();
            manager.run_passes(function);
        },
        py::arg("function"));

    m_offline_transformations.def(
        "GenerateMappingFile",
        [](std::shared_ptr<ov::Function> function, std::string path, bool extract_names) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::GenerateMappingFile>(path, extract_names);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("path"),
        py::arg("extract_names"));
}
