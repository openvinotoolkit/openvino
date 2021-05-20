//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend_manager.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_FrontEndManager(py::module m)
{
    py::class_<ngraph::frontend::FrontEndManager,
               std::shared_ptr<ngraph::frontend::FrontEndManager>>
        fem(m, "FrontEndManager", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEndManager wraps ngraph::frontend::FrontEndManager";

    fem.def(py::init<>());

    fem.def("availableFrontEnds", &ngraph::frontend::FrontEndManager::availableFrontEnds);
    fem.def("loadByFramework",
            &ngraph::frontend::FrontEndManager::loadByFramework,
            py::arg("framework"),
            py::arg("capabilities") = ngraph::frontend::FEC_DEFAULT);
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<ngraph::frontend::FrontEnd, std::shared_ptr<ngraph::frontend::FrontEnd>> fem(
        m, "FrontEnd", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def("loadFromFile", &ngraph::frontend::FrontEnd::loadFromFile, py::arg("path"));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                ngraph::frontend::InputModel::Ptr) const>(&ngraph::frontend::FrontEnd::convert));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                std::shared_ptr<ngraph::Function>) const>(&ngraph::frontend::FrontEnd::convert));
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<ngraph::frontend::Place, std::shared_ptr<ngraph::frontend::Place>> place(
        m, "Place", py::dynamic_attr());
    place.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";

    place.def("isInput", &ngraph::frontend::Place::isInput);
    place.def("isOutput", &ngraph::frontend::Place::isOutput);
    place.def("getNames", &ngraph::frontend::Place::getNames);
    place.def("isEqual", &ngraph::frontend::Place::isEqual);
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<ngraph::frontend::InputModel, std::shared_ptr<ngraph::frontend::InputModel>> im(
        m, "InputModel", py::dynamic_attr());
    im.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";
    im.def("extractSubgraph", &ngraph::frontend::InputModel::extractSubgraph);
    im.def("getPlaceByTensorName", &ngraph::frontend::InputModel::getPlaceByTensorName);
    im.def("setPartialShape", &ngraph::frontend::InputModel::setPartialShape);
    im.def("getPartialShape", &ngraph::frontend::InputModel::getPartialShape);
    im.def("getInputs", &ngraph::frontend::InputModel::getInputs);
    im.def("getOutputs", &ngraph::frontend::InputModel::getOutputs);
    im.def("overrideAllInputs", &ngraph::frontend::InputModel::overrideAllInputs);
    im.def("overrideAllOutputs", &ngraph::frontend::InputModel::overrideAllOutputs);
    im.def("setElementType", &ngraph::frontend::InputModel::setElementType);
}

void regclass_pyngraph_FEC(py::module m)
{
    py::class_<ngraph::frontend::FrontEndCapabilities,
               std::shared_ptr<ngraph::frontend::FrontEndCapabilities>>
        type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = ngraph::frontend::FEC_DEFAULT;
    type.attr("CUT") = ngraph::frontend::FEC_CUT;
    type.attr("NAMES") = ngraph::frontend::FEC_NAMES;
    type.attr("REPLACE") = ngraph::frontend::FEC_REPLACE;
    type.attr("TRAVERSE") = ngraph::frontend::FEC_TRAVERSE;
    type.attr("WILDCARDS") = ngraph::frontend::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const ngraph::frontend::FrontEndCapabilities& a,
           const ngraph::frontend::FrontEndCapabilities& b) { return a == b; },
        py::is_operator());
}

void regclass_pyngraph_GeneralFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::GeneralFailure> exc(std::move(m), "GeneralFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::GeneralFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_OpValidationFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::OpValidationFailure> exc(std::move(m),
                                                                    "OpValidationFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::OpValidationFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_OpConversionFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::OpConversionFailure> exc(std::move(m),
                                                                    "OpConversionFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::OpConversionFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_InitializationFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::InitializationFailure> exc(std::move(m),
                                                                      "InitializationFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::InitializationFailure& e)
        {
            exc(e.what());
        }
    });
}

void regclass_pyngraph_NotImplementedFailureFrontEnd(py::module m)
{
    static py::exception<ngraph::frontend::NotImplementedFailure> exc(std::move(m),
                                                                      "NotImplementedFailure");
    py::register_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const ngraph::frontend::NotImplementedFailure& e)
        {
            exc(e.what());
        }
    });
}