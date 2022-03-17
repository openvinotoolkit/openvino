// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/transformations.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/convert_fp32_to_fp16.hpp>
#include <openvino/pass/low_latency.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/pass/pass.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>

void regclass_transformations(py::module m) {
    py::class_<ov::pass::Serialize, std::shared_ptr<ov::pass::Serialize>, ov::pass::ModelPass, ov::pass::PassBase>
        serialize(m, "Serialize");
    serialize.doc() = "openvino.runtime.passes.Serialize transformation";
    serialize.def(py::init([](const std::string& path_to_xml, const std::string& path_to_bin) {
        return std::make_shared<ov::pass::Serialize>(path_to_xml, path_to_bin);
    }));
    serialize.def(py::init(
        [](const std::string& path_to_xml, const std::string& path_to_bin, ov::pass::Serialize::Version version) {
            return std::make_shared<ov::pass::Serialize>(path_to_xml, path_to_bin, version);
        }));

    py::class_<ov::pass::ConstantFolding,
               std::shared_ptr<ov::pass::ConstantFolding>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        cf(m, "ConstantFolding");
    cf.doc() = "openvino.runtime.passes.ConstantFolding transformation";
    cf.def(py::init<>());

    py::class_<ov::pass::VisualizeTree,
               std::shared_ptr<ov::pass::VisualizeTree>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        visualize(m, "VisualizeTree");
    visualize.doc() = "openvino.runtime.passes.VisualizeTree transformation";
    visualize.def(py::init<const std::string&, ov::pass::VisualizeTree::node_modifiers_t, bool>(),
                  py::arg("file_name"),
                  py::arg("nm") = nullptr,
                  py::arg("don_only") = false);

    py::class_<ov::pass::MakeStateful, std::shared_ptr<ov::pass::MakeStateful>, ov::pass::ModelPass, ov::pass::PassBase>
        make_stateful(m, "MakeStateful");
    make_stateful.doc() = "openvino.runtime.passes.MakeStateful transformation";
    make_stateful.def(py::init<const ov::pass::MakeStateful::ParamResPairs&>(), py::arg("pairs_to_replace"));
    make_stateful.def(py::init<const std::map<std::string, std::string>&>());

    py::class_<ov::pass::LowLatency2, std::shared_ptr<ov::pass::LowLatency2>, ov::pass::ModelPass, ov::pass::PassBase>
        low_latency(m, "LowLatency2");
    low_latency.doc() = "openvino.runtime.passes.LowLatency2 transformation";
    low_latency.def(py::init<bool>(), py::arg("use_const_initializer") = true);

    py::class_<ov::pass::ConvertFP32ToFP16,
               std::shared_ptr<ov::pass::ConvertFP32ToFP16>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        convert(m, "ConvertFP32ToFP16");
    convert.doc() = "openvino.runtime.passes.ConvertFP32ToFP16 transformation";
    convert.def(py::init<>());
}
