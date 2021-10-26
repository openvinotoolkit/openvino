// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "frontend_manager.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/extension.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_FrontEnd(py::module m) {
    py::class_<ngraph::frontend::FrontEnd, std::shared_ptr<ngraph::frontend::FrontEnd>> fem(m,
                                                                                            "FrontEnd",
                                                                                            py::dynamic_attr(),
                                                                                            py::module_local());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def(
        "load",
        [](ngraph::frontend::FrontEnd& self, const std::string& s) {
            return self.load(s);
        },
        py::arg("path"),
        R"(
                Loads an input model by specified model file path.

                Parameters
                ----------
                path : str
                    Main model file path.

                Returns
                ----------
                load : InputModel
                    Loaded input model.
             )");

    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                ngraph::frontend::InputModel::Ptr) const>(&ngraph::frontend::FrontEnd::convert),
            py::arg("model"),
            R"(
                Completely convert and normalize entire function, throws if it is not possible.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fem.def("convert",
            static_cast<void (ngraph::frontend::FrontEnd::*)(std::shared_ptr<ngraph::Function>) const>(
                &ngraph::frontend::FrontEnd::convert),
            py::arg("function"),
            R"(
                Completely convert the remaining, not converted part of a function.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.

                Returns
                ----------
                convert : Function
                    Fully converted nGraph function.
             )");

    fem.def("convert_partially",
            &ngraph::frontend::FrontEnd::convert_partially,
            py::arg("model"),
            R"(
                Convert only those parts of the model that can be converted leaving others as-is.
                Converted parts are not normalized by additional transformations; normalize function or
                another form of convert function should be called to finalize the conversion process.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert_partially : Function
                    Partially converted nGraph function.
             )");

    fem.def("decode",
            &ngraph::frontend::FrontEnd::decode,
            py::arg("model"),
            R"(
                Convert operations with one-to-one mapping with decoding nodes.
                Each decoding node is an nGraph node representing a single FW operation node with
                all attributes represented in FW-independent way.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                decode : Function
                    nGraph function after decoding.
             )");

    fem.def("normalize",
            &ngraph::frontend::FrontEnd::normalize,
            py::arg("function"),
            R"(
                Runs normalization passes on function that was loaded with partial conversion.

                Parameters
                ----------
                function : Function
                    Partially converted nGraph function.
             )");

    fem.def("get_name",
            &ngraph::frontend::FrontEnd::get_name,
            R"(
                Gets name of this FrontEnd. Can be used by clients
                if frontend is selected automatically by FrontEndManager::load_by_model.

                Parameters
                ----------
                get_name : str
                    Current frontend name. Empty string if not implemented.
            )");

    fem.def("add_extension",
            static_cast<void (ngraph::frontend::FrontEnd::*)(const std::shared_ptr<ov::BaseExtension>& extension)>(
                    &ngraph::frontend::FrontEnd::add_extension));

    fem.def("__repr__", [](const ngraph::frontend::FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}

void regclass_pyngraph_JsonConfigExtension(py::module m) {
    // TODO: Consider mapping of ov::Extension class instead of ov::BaseExtension in the final solution.
    // ov::BaseExtension is used now because it eliminates another level of indirection in object definitions and
    // we need less code.
    py::class_<ov::BaseExtension, std::shared_ptr<ov::BaseExtension>> ext1(m, "Extension", py::dynamic_attr());
    py::class_<ngraph::frontend::JsonConfigExtension, std::shared_ptr<ngraph::frontend::JsonConfigExtension>, ov::BaseExtension> ext2(m,
                                                                                            "JsonConfigExtension",
                                                                                            py::dynamic_attr());
    ext2.doc() = "Extension class to load and process ModelOptimier JSON config file";

    ext2.def(py::init([](const std::string& path) {
        return std::make_shared<ngraph::frontend::JsonConfigExtension>(path);
    }));
}

void regclass_pyngraph_TelemetryExtension(py::module m) {
    {
        py::class_<
                ngraph::frontend::TelemetryExtension,
                std::shared_ptr<ngraph::frontend::TelemetryExtension>,
                ov::BaseExtension> ext(m, "TelemetryExtension", py::dynamic_attr());

        ext.def(py::init([](const std::function<void(const std::string &)> callback) {
            return std::make_shared<ngraph::frontend::TelemetryExtension>(callback);
        }));

        ext.def("send", &ngraph::frontend::TelemetryExtension::send);
    }
    {
        py::class_<ngraph::frontend::NodeContext, std::shared_ptr<ngraph::frontend::NodeContext>> ext(m, "NodeContext", py::dynamic_attr());
        ext.def("optype", &ngraph::frontend::NodeContext::op_type);
        ext.def("get_ng_inputs", &ngraph::frontend::NodeContext::get_ng_inputs);
    }
    {
        py::class_<ngraph::frontend::OpExtension, std::shared_ptr<ngraph::frontend::OpExtension>, ov::BaseExtension> ext(m, "OpExtension", py::dynamic_attr());
        ext.def(py::init([](const std::string& optype, const std::function<ngraph::OutputVector(std::shared_ptr<ngraph::frontend::NodeContext>)> f) {
            return std::make_shared<ngraph::frontend::OpExtension>(optype, f);
        }));

    }
}