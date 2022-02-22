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

namespace {
class ManagerWrapper : public ov::pass::Manager {
public:
    ManagerWrapper() {}
    ~ManagerWrapper() {}

    void register_pass(const std::string& pass_name) {
        if (pass_name == "ConstantFolding")
            push_pass<ov::pass::ConstantFolding>();

        if (m_per_pass_validation)
            push_pass<ov::pass::Validate>();
        return;
    }

    void register_pass(const std::string& pass_name, const FilePaths& file_paths, const std::string& version) {
        if (pass_name == "Serialize") {
            push_pass<ov::pass::Serialize>(file_paths.first, file_paths.second, convert_to_version(version));
        }
        return;
    }
    void register_pass(const std::string& pass_name,
                       const std::string& xml_path,
                       const std::string& bin_path,
                       const std::string& version) {
        if (pass_name == "Serialize")
            push_pass<ov::pass::Serialize>(xml_path, bin_path, convert_to_version(version));
        return;
    }
};
}  // namespace

void regclass_graph_passes_Manager(py::module m) {
    py::class_<ManagerWrapper> manager(m, "Manager");
    manager.doc() = "openvino.runtime.passes.Manager wraps ov::pass::Manager using ManagerWrapper";

    manager.def(py::init<>());

    manager.def("set_per_pass_validation", &ManagerWrapper::set_per_pass_validation);
    manager.def("run_passes", &ManagerWrapper::run_passes);
    manager.def("register_pass",
                (void (ManagerWrapper::*)(const std::string&)) & ManagerWrapper::register_pass,
                py::arg("pass_name"),
                R"(
        Set the type of register pass for pass manager.
        Parameters
        ----------
        pass_name : str
            string to set the type of a pass
    // )");

    manager.def("register_pass",
                (void (ManagerWrapper::*)(const std::string&, const FilePaths&, const std::string&)) &
                    ManagerWrapper::register_pass,
                py::arg("pass_name"),
                py::arg("output_files"),
                py::arg("version") = "UNSPECIFIED",
                R"(
        Set the type of register pass for pass manager.
        Parameters
        ----------
        pass_name : str
            string to set the type of a pass
        output_files : Tuple[str, str]
            tuple which contains paths where .xml and .bin files will be saved
        version : str
            sets the version of the IR which will be generated.
            Supported versions are:
                            - "UNSPECIFIED" (default) : Use the latest or function version
                            - "IR_V10" : v10 IR
                            - "IR_V11" : v11 IR
        Examples
        ----------
        1. Default Version
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", output_files=("example.xml", "example.bin"))
        2. IR version 11
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", output_files=("example.xml", "example.bin"), version="IR_V11")
    // )");
    manager.def(
        "register_pass",
        (void (ManagerWrapper::*)(const std::string&, const std::string&, const std::string&, const std::string&)) &
            ManagerWrapper::register_pass,
        py::arg("pass_name"),
        py::arg("xml_path"),
        py::arg("bin_path"),
        py::arg("version") = "UNSPECIFIED",
        R"(
        Set the type of register pass for pass manager.
        Parameters
        ----------
        pass_name : str
            string to set the type of a pass
        xml_path : str
            path where .xml file will be saved
        bin_path : str
            path where .bin file will be saved
        version : str
            sets the version of the IR which will be generated.
            Supported versions are:
                            - "UNSPECIFIED" (default) : Use the latest or function version
                            - "IR_V10" : v10 IR
                            - "IR_V11" : v11 IR
        Examples
        ----------
        1. Default Version
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", xml_path="example.xml", bin_path="example.bin")
        2. IR version 11
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", xml_path="example.xml", bin_path="example.bin", version="IR_V11")
    // )");
}
