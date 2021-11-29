// Copyright (C) 2018-2021 Intel Corporation
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

namespace {
class ManagerWrapper : public ov::pass::Manager {
public:
    ManagerWrapper() {}
    ~ManagerWrapper() {}

    void register_pass(std::string pass_name, py::args args, const py::kwargs& kwargs) {
        if (pass_name == "ConstantFolding")
            push_pass<ov::pass::ConstantFolding>();

        if (m_per_pass_validation)
            push_pass<ov::pass::Validate>();

        if (pass_name == "Serialize") {
            const auto num_of_args = args.size() + kwargs.size();
            if (num_of_args == 2) {
                const auto xml_path = kwargs.contains("xml_path") ? py::cast<std::string>(kwargs["xml_path"])
                                                                  : py::cast<std::string>(args[0]);
                const auto bin_path = kwargs.contains("bin_path") ? py::cast<std::string>(kwargs["bin_path"])
                                                                  : py::cast<std::string>(args[1]);
                push_pass<ov::pass::Serialize>(xml_path, bin_path, Version::UNSPECIFIED);
            } else if (num_of_args == 1) {
                const auto file_paths = kwargs.contains("output_files") ? py::cast<FilePaths>(kwargs["output_files"])
                                                                        : py::cast<FilePaths>(args[0]);
                push_pass<ov::pass::Serialize>(file_paths.first, file_paths.second, Version::UNSPECIFIED);
            } else {
                throw ov::Exception("Invoked with wrong number of arguments! Please be sure to provide paths where "
                                    "generated files will be saved either as two strings or as a tuple(str, str).");
            }
        }
        return;
    }
};
}  // namespace

void regclass_graph_passes_Manager(py::module m) {
    py::class_<ManagerWrapper> manager(m, "Manager");
    manager.doc() = "openvino.impl.passes.Manager wraps ov::pass::Manager using ManagerWrapper";

    manager.def(py::init<>());

    manager.def("set_per_pass_validation", &ManagerWrapper::set_per_pass_validation);
    manager.def("run_passes", &ManagerWrapper::run_passes);
    manager.def("register_pass",
                &ManagerWrapper::register_pass,
                py::arg("pass_name"),
                R"(
        Set the type of register pass for pass manager.

        Parameters
        ----------
        pass_name : str
            string to set the type of a pass

        Kwargs:
            Paths where generated files will be saved
            can be provided either via two seperate arguments,
            namely xml_path and bin path, or as a single tuple
            output_files.

            xml_path : str
                path where .xml file will be saved
            bin_path : str
                path where .bin file will be saved
            output_files : Tuple[str, str]
                tuple which contains paths where .xml and .bin files will be saved

        Examples:
        ----------
        1. Seperate paths:
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", xml_path="example.xml", bin_path="example.bin")

        2. Tuple containing paths:
            pass_manager = Manager()
            pass_manager.register_pass("Serialize", output_files=("example.xml", "example.bin"))
    // )");
}
