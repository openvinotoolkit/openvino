// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

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
                ngraph::frontend::InputModel::Ptr) const>(&ngraph::frontend::FrontEnd::convert));
    fem.def("convert",
            static_cast<std::shared_ptr<ngraph::Function> (ngraph::frontend::FrontEnd::*)(
                std::shared_ptr<ngraph::Function>) const>(&ngraph::frontend::FrontEnd::convert));
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
              &ngraph::frontend::Place::get_consuming_operations,
              py::arg_v("outputPortIndex", -1, "-1"));
    place.def("get_target_tensor",
              &ngraph::frontend::Place::get_target_tensor,
              py::arg_v("outputPortIndex", -1, "-1"));
    place.def("get_producing_operation",
              &ngraph::frontend::Place::get_producing_operation,
              py::arg_v("inputPortIndex", -1, "-1"));
    place.def("get_producing_port", &ngraph::frontend::Place::get_producing_port);
    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_input_port),
              py::arg_v("inputPortIndex", -1, "-1"));
    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&, int) const>(&ngraph::frontend::Place::get_input_port),
              py::arg("inputName"),
              py::arg_v("inputPortIndex", -1, "-1"));
    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_output_port),
              py::arg_v("outputPortIndex", -1, "-1"));
    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&, int) const>(&ngraph::frontend::Place::get_output_port),
              py::arg("outputName"),
              py::arg_v("outputPortIndex", -1, "-1"));
    place.def("get_consuming_ports", &ngraph::frontend::Place::get_consuming_ports);
    place.def("get_source_tensor",
              &ngraph::frontend::Place::get_source_tensor,
              py::arg_v("inputPortIndex", -1, "-1"));
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<ngraph::frontend::InputModel, std::shared_ptr<ngraph::frontend::InputModel>> im(
        m, "InputModel", py::dynamic_attr());
    im.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";
    im.def("extract_subgraph", &ngraph::frontend::InputModel::extract_subgraph);
    im.def("get_place_by_tensor_name", &ngraph::frontend::InputModel::get_place_by_tensor_name);
    im.def("get_place_by_operation_name",
           &ngraph::frontend::InputModel::get_place_by_operation_name);
    im.def("get_place_by_operation_and_input_port",
           &ngraph::frontend::InputModel::get_place_by_operation_and_input_port);
    im.def("get_place_by_operation_and_output_port",
           &ngraph::frontend::InputModel::get_place_by_operation_and_output_port);

    im.def("set_name_for_tensor", &ngraph::frontend::InputModel::set_name_for_tensor);
    im.def("add_name_for_tensor", &ngraph::frontend::InputModel::add_name_for_tensor);
    im.def("set_name_for_operation", &ngraph::frontend::InputModel::set_name_for_operation);
    im.def("free_name_for_tensor", &ngraph::frontend::InputModel::free_name_for_tensor);
    im.def("free_name_for_operation", &ngraph::frontend::InputModel::free_name_for_operation);
    im.def("set_name_for_dimension", &ngraph::frontend::InputModel::set_name_for_dimension);
    im.def("cut_and_add_new_input", &ngraph::frontend::InputModel::cut_and_add_new_input);
    im.def("cut_and_add_new_output", &ngraph::frontend::InputModel::cut_and_add_new_output);
    im.def("add_output", &ngraph::frontend::InputModel::add_output);
    im.def("remove_output", &ngraph::frontend::InputModel::remove_output);

    // Setting tensor properties
    im.def("set_partial_shape", &ngraph::frontend::InputModel::set_partial_shape);
    im.def("get_partial_shape", &ngraph::frontend::InputModel::get_partial_shape);
    im.def("get_inputs", &ngraph::frontend::InputModel::get_inputs);
    im.def("get_outputs", &ngraph::frontend::InputModel::get_outputs);
    im.def("override_all_inputs", &ngraph::frontend::InputModel::override_all_inputs);
    im.def("override_all_outputs", &ngraph::frontend::InputModel::override_all_outputs);
    im.def("set_element_type", &ngraph::frontend::InputModel::set_element_type);
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
