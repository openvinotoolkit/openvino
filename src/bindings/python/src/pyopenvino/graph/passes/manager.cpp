// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/validate.hpp"
#include "pyopenvino/graph/passes/manager.hpp"

namespace py = pybind11;

using Version = ov::pass::Serialize::Version;
using FilePaths = std::pair<const std::string, const std::string>;

inline Version convert_to_version(const std::string& version) {
    if (version == "UNSPECIFIED")
        return Version::UNSPECIFIED;
    if (version == "IR_V10")
        return Version::IR_V10;
    if (version == "IR_V11")
        return Version::IR_V11;
    throw ov::Exception("Invoked with wrong version argument: '" + version +
                        "'! The supported versions are: 'UNSPECIFIED'(default), 'IR_V10', 'IR_V11'.");
}

void regclass_passes_Manager(py::module m) {
    py::class_<ov::pass::Manager> manager(m, "Manager");
    manager.doc() = "openvino.runtime.passes.Manager executes sequence of transformation on a given Model";

    manager.def(py::init<>());
    manager.def("set_per_pass_validation",
                &ov::pass::Manager::set_per_pass_validation,
                py::arg("new_state"),
                R"(
                Enables or disables Model validation after each pass execution.

                :param new_state: flag which enables or disables model validation.
                :type new_state: bool
    )");

    manager.def("run_passes",
                &ov::pass::Manager::run_passes,
                py::arg("model"),
                R"(
                Executes sequence of transformations on given Model.

                :param model: openvino.runtime.Model to be transformed.
                :type model: openvino.runtime.Model
    )");

    manager.def("register_pass",
                &ov::pass::Manager::register_pass_instance,
                py::arg("transformation"),
                R"(
                Register pass instance for execution. Execution order matches the registration order.

                :param transformation: transformation instance.
                :type transformation: openvino.runtime.passes.PassBase
    )");

    manager.def(
        "register_pass",
        [](ov::pass::Manager& self, const std::string& pass_name) -> void {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "register_pass with this arguments is deprecated! "
                         "Please use register_pass(ConstantFolding()) instead.",
                         1);
            if (pass_name == "ConstantFolding") {
                self.register_pass<ov::pass::ConstantFolding>();
            }
        },
        py::arg("pass_name"),
        R"(
                This method is deprecated. Please use m.register_pass(ConstantFolding()) instead.

                Register pass by name from the list of predefined passes.

                :param pass_name: String to set the type of a pass.
                :type pass_name: str
    )");

    manager.def(
        "register_pass",
        [](ov::pass::Manager& self,
           const std::string& pass_name,
           const FilePaths& file_paths,
           const std::string& version) -> void {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "register_pass with this arguments is deprecated! "
                         "Please use register_pass(Serialize(xml, bin, version)) instead.",
                         1);
            if (pass_name == "Serialize") {
                self.register_pass<ov::pass::Serialize>(file_paths.first,
                                                        file_paths.second,
                                                        convert_to_version(version));
            }
        },
        py::arg("pass_name"),
        py::arg("output_files"),
        py::arg("version") = "UNSPECIFIED",
        R"(
        This method is deprecated. Please use m.register_pass(Serialize(...)) instead.

        Set the type of register pass for pass manager.

        :param pass_name: String to set the type of a pass.
        :type pass_name: str
        :param output_files: Tuple which contains paths where .xml and .bin files will be saved.
        :type output_files: Tuple[str, str]
        :param version: Sets the version of the IR which will be generated.
                                   Supported versions are:
                                       - "UNSPECIFIED" (default) : Use the latest or function version
                                       - "IR_V10" : v10 IR
                                       - "IR_V11" : v11 IR
        :type version: str

        Examples
        ----------
        1. Default Version
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", output_files=("example.xml", "example.bin"))
        2. IR version 11
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", output_files=("example.xml", "example.bin"), version="IR_V11")
    )");

    manager.def(
        "register_pass",
        [](ov::pass::Manager& self,
           const std::string& pass_name,
           const std::string& xml_path,
           const std::string& bin_path,
           const std::string& version) -> void {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "register_pass with this arguments is deprecated! "
                         "Please use register_pass(Serialize(xml, bin, version)) instead.",
                         1);
            if (pass_name == "Serialize") {
                self.register_pass<ov::pass::Serialize>(xml_path, bin_path, convert_to_version(version));
            }
        },
        py::arg("pass_name"),
        py::arg("xml_path"),
        py::arg("bin_path"),
        py::arg("version") = "UNSPECIFIED",
        R"(
        This method is deprecated. Please use m.register_pass(Serialize(...)) instead.

        Set the type of register pass for pass manager.

        :param pass_name: String to set the type of a pass.
        :type pass_name: str
        :param xml_path: Path where *.xml file will be saved.
        :type xml_path: str
        :param bin_path: Path where *.bin file will be saved.
        :type bin_path: str
        :param version: Sets the version of the IR which will be generated.
            Supported versions are:
                            - "UNSPECIFIED" (default) : Use the latest or function version
                            - "IR_V10" : v10 IR
                            - "IR_V11" : v11 IR
        :type version: str

        Examples
        ----------
        1. Default Version
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", xml_path="example.xml", bin_path="example.bin")
        2. IR version 11
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", xml_path="example.xml", bin_path="example.bin", version="IR_V11")
    )");
}
