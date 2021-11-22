// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/validate.hpp"
#include "pyopenvino/graph/passes/manager.hpp"

namespace py = pybind11;


// TODO:
// better expections
// better docs
namespace {
class ManagerWrapper : public ov::pass::Manager {
public:
    ManagerWrapper() {}
    ~ManagerWrapper() {}

    // void register_pass(std::string pass_name, py::args args, const py::kwargs& kwargs) {
    //     if (pass_name == "ConstantFolding")
    //         push_pass<ov::pass::ConstantFolding>();

    //     if (m_per_pass_validation)
    //         push_pass<ov::pass::Validate>();

    //     if (pass_name == "Serialize") {
    //         const auto args_size = args.size();
    //         const auto kwargs_size = kwargs.size();
    //         if ((args_size > 0) || (kwargs_size > 0)) {
    //             if (args_size + kwargs_size > 2) {
    //                 throw ov::Exception("Invoked with too many arguments! The possible args are: xml_path and
    //                 bin_path");
    //             }
    //             const auto xml_path = kwargs.contains("xml_path") ? py::cast<std::string>(kwargs["xml_path"]) :
    //             py::cast<std::string>(args[0]); const auto bin_path = kwargs.contains("bin_path") ?
    //             py::cast<std::string>(kwargs["bin_path"]) : py::cast<std::string>(args[1]);
    //             push_pass<ov::pass::Serialize>(xml_path, bin_path, Version::UNSPECIFIED);
    //         } else {
    //              throw ov::Exception("No required paths specified for saving .xml and .bin files!");
    //         }
    //     }
    //     return;
    // }

    void register_pass(std::string pass_name, const std::string& xml_path, const std::string& bin_path) {
        if (pass_name == "ConstantFolding")
            push_pass<ov::pass::ConstantFolding>();

        if (m_per_pass_validation)
            push_pass<ov::pass::Validate>();

        if (pass_name == "Serialize") {
            if ((xml_path.size() >= 4) && (bin_path.size() >= 4))
                push_pass<ov::pass::Serialize>(xml_path, bin_path, Version::UNSPECIFIED);
            else
                throw ov::Exception("One or more required file paths for saving .xml and .bin files are incorrect!");
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

    // TODO: better docs
    // manager.def(
    //     "register_pass",
    //     (void (ManagerWrapper::*)(std::string, py::args args, const py::kwargs& kwargs)) &
    //         ManagerWrapper::register_pass,
    // py::arg("pass_name"),
    // R"(
    //     Set the type of register pass for pass manager.
    //     Both optional parameters are related to
    //     'Serialize' pass.

    //     Parameters
    //     ----------
    //     pass_name : str
    //         String to set the type of pass
    //     Args:
    //         param1 : str
    //             path where .xml file will be saved
    //         param2 : str
    //             path where .bin file will be saved

    //     Kwargs:
    //         xml_path : str
    //             path where .xml file will be saved
    //         bin_path : str
    //             path where .bin file will be saved
    // )");

    manager.def(
        // TODO: better docs
        "register_pass",
        (void (ManagerWrapper::*)(std::string, const std::string&, const std::string)) & ManagerWrapper::register_pass,
        py::arg("pass_name"),
        py::arg("xml_path") = "",
        py::arg("bin_path") = "",
        R"(
        Set the type of register pass for pass manager.
        Both optional parameters are related to
        'Serialize' pass. 

        Parameters
        ----------
        pass_name : str
            String to set the type of pass
        xml_path  : Optional[str]
            path where .xml file will be save
        bin_path  : Optional[str]
            path where .bin file will be save

    )");
}
