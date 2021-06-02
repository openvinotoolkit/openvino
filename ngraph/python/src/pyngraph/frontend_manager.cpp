// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    fem.def("get_available_front_ends",
            &ngraph::frontend::FrontEndManager::get_available_front_ends);
    fem.def("load_by_framework",
            &ngraph::frontend::FrontEndManager::load_by_framework,
            py::arg("framework"),
            py::arg("capabilities") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT);
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<ngraph::frontend::FrontEnd, std::shared_ptr<ngraph::frontend::FrontEnd>> fem(
        m, "FrontEnd", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def("load_from_file", &ngraph::frontend::FrontEnd::load_from_file, py::arg("path"));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                ngraph::frontend::InputModel::Ptr) const>(&ngraph::frontend::FrontEnd::convert),
            py::arg("model"));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                std::shared_ptr<ngraph::Function>) const>(&ngraph::frontend::FrontEnd::convert),
            py::arg("function"));
    fem.def("convert_partially", &ngraph::frontend::FrontEnd::convert_partially, py::arg("model"));
    fem.def("decode", &ngraph::frontend::FrontEnd::decode, py::arg("model"));
    fem.def("normalize", &ngraph::frontend::FrontEnd::normalize, py::arg("function"));
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<ngraph::frontend::Place, std::shared_ptr<ngraph::frontend::Place>> place(
        m, "Place", py::dynamic_attr());
    place.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";

    place.def("is_input", &ngraph::frontend::Place::is_input);
    place.def("is_output", &ngraph::frontend::Place::is_output);
    place.def("get_names", &ngraph::frontend::Place::get_names);
    place.def("is_equal", &ngraph::frontend::Place::is_equal, py::arg("other"));
    place.def("is_equal_data", &ngraph::frontend::Place::is_equal_data, py::arg("other"));

    place.def("get_consuming_operations",
              static_cast<std::vector<ngraph::frontend::Place::Ptr>(ngraph::frontend::Place::*)() const>(
                      &ngraph::frontend::Place::get_consuming_operations));
    place.def("get_consuming_operations",
              static_cast<std::vector<ngraph::frontend::Place::Ptr>(ngraph::frontend::Place::*)(int) const>(
                      &ngraph::frontend::Place::get_consuming_operations),
              py::arg("outputPortIndex"));

    place.def("get_target_tensor",
              static_cast<ngraph::frontend::Place::Ptr(ngraph::frontend::Place::*)() const>(
                      &ngraph::frontend::Place::get_target_tensor));
    place.def("get_target_tensor",
              static_cast<ngraph::frontend::Place::Ptr(ngraph::frontend::Place::*)(int) const>(
                      &ngraph::frontend::Place::get_target_tensor),
              py::arg("outputPortIndex"));

    place.def("get_producing_operation",
              static_cast<ngraph::frontend::Place::Ptr(ngraph::frontend::Place::*)() const>(
                      &ngraph::frontend::Place::get_producing_operation));
    place.def("get_producing_operation",
              static_cast<ngraph::frontend::Place::Ptr(ngraph::frontend::Place::*)(int) const>(
                      &ngraph::frontend::Place::get_producing_operation),
              py::arg("inputPortIndex"));

    place.def("get_producing_port", &ngraph::frontend::Place::get_producing_port);

    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                      &ngraph::frontend::Place::get_input_port));
    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_input_port),
              py::arg("inputPortIndex"));
    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                      const std::string&) const>(&ngraph::frontend::Place::get_input_port),
              py::arg("inputName"));
    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&, int) const>(&ngraph::frontend::Place::get_input_port),
              py::arg("inputName"),
              py::arg("inputPortIndex"));

    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_output_port));
    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                      &ngraph::frontend::Place::get_output_port),
              py::arg("outputPortIndex"));
    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&) const>(&ngraph::frontend::Place::get_output_port),
              py::arg("outputName"));
    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                      const std::string&, int) const>(&ngraph::frontend::Place::get_output_port),
              py::arg("outputName"),
              py::arg("outputPortIndex"));

    place.def("get_consuming_ports", &ngraph::frontend::Place::get_consuming_ports);

    place.def("get_source_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                      &ngraph::frontend::Place::get_source_tensor));
    place.def("get_source_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                      &ngraph::frontend::Place::get_source_tensor),
              py::arg("inputPortIndex"));
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<ngraph::frontend::InputModel, std::shared_ptr<ngraph::frontend::InputModel>> im(
        m, "InputModel", py::dynamic_attr());
    im.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";
    im.def("get_place_by_tensor_name",
           &ngraph::frontend::InputModel::get_place_by_tensor_name,
           py::arg("tensorName"));
    im.def("get_place_by_operation_name",
           &ngraph::frontend::InputModel::get_place_by_operation_name,
           py::arg("operationName"));
    im.def("get_place_by_operation_name_and_input_port",
           &ngraph::frontend::InputModel::get_place_by_operation_name_and_input_port,
           py::arg("operationName"),
           py::arg("inputPortIndex"));
    im.def("get_place_by_operation_name_and_output_port",
           &ngraph::frontend::InputModel::get_place_by_operation_name_and_output_port,
           py::arg("operationName"),
           py::arg("outputPortIndex"));

    im.def("set_name_for_tensor",
           &ngraph::frontend::InputModel::set_name_for_tensor,
           py::arg("tensor"),
           py::arg("newName"));
    im.def("add_name_for_tensor",
           &ngraph::frontend::InputModel::add_name_for_tensor,
           py::arg("tensor"),
           py::arg("newName"));
    im.def("set_name_for_operation",
           &ngraph::frontend::InputModel::set_name_for_operation,
           py::arg("operation"),
           py::arg("newName"));
    im.def("free_name_for_tensor",
           &ngraph::frontend::InputModel::free_name_for_tensor,
           py::arg("name"));
    im.def("free_name_for_operation",
           &ngraph::frontend::InputModel::free_name_for_operation,
           py::arg("name"));
    im.def("set_name_for_dimension",
           &ngraph::frontend::InputModel::set_name_for_dimension,
           py::arg("place"),
           py::arg("dimIndex"),
           py::arg("dimName"));
    im.def("cut_and_add_new_input",
           &ngraph::frontend::InputModel::cut_and_add_new_input,
           py::arg("place"),
           py::arg("newName") = std::string());
    im.def("cut_and_add_new_output",
           &ngraph::frontend::InputModel::cut_and_add_new_output,
           py::arg("place"),
           py::arg("newName") = std::string());
    im.def("add_output", &ngraph::frontend::InputModel::add_output, py::arg("place"));
    im.def("remove_output", &ngraph::frontend::InputModel::remove_output, py::arg("place"));

    im.def("set_partial_shape",
           &ngraph::frontend::InputModel::set_partial_shape,
           py::arg("place"),
           py::arg("shape"));
    im.def("get_partial_shape", &ngraph::frontend::InputModel::get_partial_shape, py::arg("place"));
    im.def("get_inputs", &ngraph::frontend::InputModel::get_inputs);
    im.def("get_outputs", &ngraph::frontend::InputModel::get_outputs);

    im.def("extract_subgraph",
           &ngraph::frontend::InputModel::extract_subgraph,
           py::arg("inputs"),
           py::arg("outputs"));
    im.def("override_all_inputs",
           &ngraph::frontend::InputModel::override_all_inputs,
           py::arg("inputs"));
    im.def("override_all_outputs",
           &ngraph::frontend::InputModel::override_all_outputs,
           py::arg("outputs"));
    im.def("set_element_type",
           &ngraph::frontend::InputModel::set_element_type,
           py::arg("place"),
           py::arg("type"));
}

void regclass_pyngraph_FEC(py::module m)
{
    class FeCaps
    {
    public:
        int get_caps() const { return m_caps; }

    private:
        int m_caps;
    };

    py::class_<FeCaps, std::shared_ptr<FeCaps>> type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT;
    type.attr("CUT") = ngraph::frontend::FrontEndCapabilities::FEC_CUT;
    type.attr("NAMES") = ngraph::frontend::FrontEndCapabilities::FEC_NAMES;
    type.attr("WILDCARDS") = ngraph::frontend::FrontEndCapabilities::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const FeCaps& a, const FeCaps& b) { return a.get_caps() == b.get_caps(); },
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