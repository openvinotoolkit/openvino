// Copyright (C) 2018-2025 Intel Corporation
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

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;
using Version = ov::pass::Serialize::Version;

void regclass_transformations(py::module m) {
    py::enum_<Version>(m, "Version", py::arithmetic())
        .value("UNSPECIFIED", Version::UNSPECIFIED)
        .value("IR_V10", Version::IR_V10)
        .value("IR_V11", Version::IR_V11);

    py::class_<ov::pass::Serialize, std::shared_ptr<ov::pass::Serialize>, ov::pass::ModelPass, ov::pass::PassBase>
        serialize(m, "Serialize");
    serialize.doc() = "openvino.passes.Serialize transformation";

    serialize.def(
        py::init([](const py::object& path_to_xml, const py::object& path_to_bin, const py::object& version) {
            if (py::isinstance<py::str>(version)) {
                return std::make_shared<ov::pass::Serialize>(
                    Common::utils::convert_path_to_string(path_to_xml),
                    Common::utils::convert_path_to_string(path_to_bin),
                    Common::utils::convert_to_version(version.cast<std::string>()));
            } else if (py::isinstance<Version>(version)) {
                return std::make_shared<ov::pass::Serialize>(Common::utils::convert_path_to_string(path_to_xml),
                                                             Common::utils::convert_path_to_string(path_to_bin),
                                                             version.cast<Version>());
            } else {
                return std::make_shared<ov::pass::Serialize>(Common::utils::convert_path_to_string(path_to_xml),
                                                             Common::utils::convert_path_to_string(path_to_bin));
            }
        }),
        py::arg("path_to_xml"),
        py::arg("path_to_bin"),
        py::arg("version") = py::none(),
        R"(
        Create Serialize pass which is used for Model to IR serialization.

        :param path_to_xml: Path where *.xml file will be saved.
        :type path_to_xml: Union[str, bytes, pathlib.Path]

        :param path_to_xml: Path where *.bin file will be saved.
        :type path_to_xml: Union[str, bytes, pathlib.Path]

        :param version: Optional serialized IR version.
        :type version: Union[str, openvino.passes.Version]
    )");

    serialize.def("__repr__", [](const ov::pass::Serialize& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::pass::ConstantFolding,
               std::shared_ptr<ov::pass::ConstantFolding>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        cf(m, "ConstantFolding");
    cf.doc() = "openvino.passes.ConstantFolding transformation";
    cf.def(py::init<>());
    cf.def("__repr__", [](const ov::pass::ConstantFolding& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::pass::VisualizeTree,
               std::shared_ptr<ov::pass::VisualizeTree>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        visualize(m, "VisualizeTree");
    visualize.doc() = "openvino.passes.VisualizeTree transformation";
    visualize.def(py::init<const std::string&, ov::pass::VisualizeTree::node_modifiers_t, bool>(),
                  py::arg("file_name"),
                  py::arg("nm") = nullptr,
                  py::arg("don_only") = false,
                  R"(
                  Create VisualizeTree pass which is used for Model to dot serialization.

                  :param file_name: Path where serialized model will be saved. For example: /tmp/out.svg
                  :type file_name: str

                  :param nm: Node modifier function.
                  :type nm: function

                  :param don_only: Enable only dot file generation.
                  :type don_only: bool
    )");
    visualize.def("__repr__", [](const ov::pass::VisualizeTree& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::pass::MakeStateful, std::shared_ptr<ov::pass::MakeStateful>, ov::pass::ModelPass, ov::pass::PassBase>
        make_stateful(m, "MakeStateful");
    make_stateful.doc() = "openvino.passes.MakeStateful transformation";
    make_stateful.def(
        py::init<const ov::pass::MakeStateful::ParamResPairs&>(),
        py::arg("pairs_to_replace"),
        R"( The transformation replaces the provided pairs Parameter and Result with openvino Memory operations ReadValue and Assign.
                    
                      :param pairs_to_replace:
                      :type pairs_to_replace: List[Tuple[op.Parameter, op.Result]
    )");
    make_stateful.def(py::init<const std::map<std::string, std::string>&>(),
                      py::arg("pairs_to_replace"),
                      R"(
        The transformation replaces the provided pairs Parameter and Result with openvino Memory operations ReadValue and Assign.
        
        :param pairs_to_replace: a dictionary of names of the provided Parameter and Result operations.
        :type pairs_to_replace: Dict[str, str]
    )");
    make_stateful.def("__repr__", [](const ov::pass::MakeStateful& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::pass::LowLatency2, std::shared_ptr<ov::pass::LowLatency2>, ov::pass::ModelPass, ov::pass::PassBase>
        low_latency(m, "LowLatency2");
    low_latency.doc() = "openvino.passes.LowLatency2 transformation";

    low_latency.def(py::init<bool>(),
                    py::arg("use_const_initializer") = true,
                    R"(
                    Create LowLatency2 pass which is used for changing the structure of the model,
                    which contains TensorIterator/Loop operations.
                    The transformation finds all TensorIterator/Loop layers in the network, 
                    processes all back edges that describe a connection between Result and Parameter of the TensorIterator/Loop bodies, 
                    and inserts ReadValue and Assign layers at the input and output corresponding to this back edge.

                    :param use_const_initializer: Changes the type of the initializing subgraph for ReadValue operations.
                                                  If "true", then the transformation inserts Constant before ReadValue operation.
                                                  If "false, then the transformation leaves existed initializing subgraph for ReadValue operation.
                    :type use_const_initializer: bool
    )");
    low_latency.def("__repr__", [low_latency](const ov::pass::LowLatency2& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::pass::ConvertFP32ToFP16,
               std::shared_ptr<ov::pass::ConvertFP32ToFP16>,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        convert(m, "ConvertFP32ToFP16");
    convert.doc() = "openvino.passes.ConvertFP32ToFP16 transformation";
    convert.def(py::init<>());
    convert.def("__repr__", [](const ov::pass::ConvertFP32ToFP16& self) {
        return Common::get_simple_repr(self);
    });
}
