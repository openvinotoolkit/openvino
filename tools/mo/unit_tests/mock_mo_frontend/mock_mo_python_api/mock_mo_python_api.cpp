// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../mock_mo_frontend/mock_mo_frontend.hpp"

namespace py = pybind11;
using namespace ov::frontend;

static void register_mock_frontend_stat(py::module m)
{
    m.def("get_frontend_statistic", &FrontEndMockPy::get_stat);
    m.def("clear_frontend_statistic", &FrontEndMockPy::clear_stat);

    py::class_<FeStat> feStat(m, "FeStat", py::dynamic_attr());
    feStat.def_property_readonly("load_paths", &FeStat::load_paths);
    feStat.def_property_readonly("convert_model", &FeStat::convert_model);
    feStat.def_property_readonly("supported", &FeStat::supported);
    feStat.def_property_readonly("get_name", &FeStat::get_name);
}

static void register_mock_setup(py::module m)
{
    m.def("clear_setup", &MockSetup::clear_setup);
    m.def("set_equal_data", &MockSetup::set_equal_data);
    m.def("set_max_port_counts", &MockSetup::set_max_port_counts);
}

static void register_mock_model_stat(py::module m)
{
    m.def("get_model_statistic", &InputModelMockPy::get_stat);
    m.def("clear_model_statistic", &InputModelMockPy::clear_stat);
    m.def("mock_return_partial_shape", &InputModelMockPy::mock_return_partial_shape);

    py::class_<ModelStat> mdlStat(m, "ModelStat", py::dynamic_attr());
    mdlStat.def_property_readonly("get_inputs", &ModelStat::get_inputs);
    mdlStat.def_property_readonly("get_outputs", &ModelStat::get_outputs);
    mdlStat.def_property_readonly("get_place_by_operation_name",
                                  &ModelStat::get_place_by_operation_name);
    mdlStat.def_property_readonly("get_place_by_tensor_name", &ModelStat::get_place_by_tensor_name);

    mdlStat.def_property_readonly("set_partial_shape", &ModelStat::set_partial_shape);
    mdlStat.def_property_readonly("get_partial_shape", &ModelStat::get_partial_shape);
    mdlStat.def_property_readonly("set_element_type", &ModelStat::set_element_type);
    mdlStat.def_property_readonly("extract_subgraph", &ModelStat::extract_subgraph);
    mdlStat.def_property_readonly("override_all_inputs", &ModelStat::override_all_inputs);
    mdlStat.def_property_readonly("override_all_outputs", &ModelStat::override_all_outputs);

    // Arguments tracking
    mdlStat.def_property_readonly("lastArgString", &ModelStat::get_lastArgString);
    mdlStat.def_property_readonly("lastArgInt", &ModelStat::get_lastArgInt);
    mdlStat.def_property_readonly("lastArgPlace", &ModelStat::get_lastArgPlace);
    mdlStat.def_property_readonly("lastArgInputPlaces", &ModelStat::get_lastArgInputPlaces);
    mdlStat.def_property_readonly("lastArgOutputPlaces", &ModelStat::get_lastArgOutputPlaces);
    mdlStat.def_property_readonly("lastArgElementType", &ModelStat::get_lastArgElementType);
    mdlStat.def_property_readonly("lastArgPartialShape", &ModelStat::get_lastArgPartialShape);
}

static void register_mock_place_stat(py::module m)
{
    m.def("get_place_statistic", &PlaceMockPy::get_stat);
    m.def("clear_place_statistic", &PlaceMockPy::clear_stat);

    py::class_<PlaceStat> placeStat(m, "PlaceStat", py::dynamic_attr());

    placeStat.def_property_readonly("lastArgString", &PlaceStat::get_lastArgString);
    placeStat.def_property_readonly("lastArgInt", &PlaceStat::get_lastArgInt);
    placeStat.def_property_readonly("lastArgPlace", &PlaceStat::get_lastArgPlace);

    placeStat.def_property_readonly("get_names", &PlaceStat::get_names);
    placeStat.def_property_readonly("get_input_port", &PlaceStat::get_input_port);
    placeStat.def_property_readonly("get_output_port", &PlaceStat::get_output_port);
    placeStat.def_property_readonly("is_input", &PlaceStat::is_input);
    placeStat.def_property_readonly("is_output", &PlaceStat::is_output);
    placeStat.def_property_readonly("is_equal", &PlaceStat::is_equal);
    placeStat.def_property_readonly("is_equal_data", &PlaceStat::is_equal_data);
}

PYBIND11_MODULE(mock_mo_python_api, m)
{
    m.doc() = "Mock frontend call counters for testing Pyngraph frontend bindings";
    register_mock_frontend_stat(m);
    register_mock_setup(m);
    register_mock_model_stat(m);
    register_mock_place_stat(m);
}
