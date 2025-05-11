// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mock_py_frontend/frontend_wrappers.hpp"
#include "mock_py_frontend/mock_py_frontend.hpp"

namespace py = pybind11;
using namespace ov::frontend;

static void register_mock_frontend_stat(py::module m) {
    m.def("get_fe_stat", &FrontEndMockPy::get_stat);
    m.def("clear_fe_stat", &FrontEndMockPy::clear_stat);

    py::class_<FeStat> feStat(m, "FeStat", py::dynamic_attr());
    feStat.def_property_readonly("load_paths", &FeStat::load_paths);
    feStat.def_property_readonly("convert_model", &FeStat::convert_model);
    feStat.def_property_readonly("convert", &FeStat::convert);
    feStat.def_property_readonly("convert_partially", &FeStat::convert_partially);
    feStat.def_property_readonly("decode", &FeStat::decode);
    feStat.def_property_readonly("normalize", &FeStat::normalize);
    feStat.def_property_readonly("get_name", &FeStat::get_name);
    feStat.def_property_readonly("supported", &FeStat::supported);
}

static void register_mock_model_stat(py::module m) {
    m.def("get_mdl_stat", &InputModelMockPy::get_stat);
    m.def("clear_mdl_stat", &InputModelMockPy::clear_stat);

    py::class_<ModelStat> mdlStat(m, "ModelStat", py::dynamic_attr());
    mdlStat.def_property_readonly("get_inputs", &ModelStat::get_inputs);
    mdlStat.def_property_readonly("get_outputs", &ModelStat::get_outputs);
    mdlStat.def_property_readonly("get_place_by_tensor_name", &ModelStat::get_place_by_tensor_name);
    mdlStat.def_property_readonly("get_place_by_operation_name", &ModelStat::get_place_by_operation_name);
    mdlStat.def_property_readonly("get_place_by_operation_and_input_port",
                                  &ModelStat::get_place_by_operation_and_input_port);
    mdlStat.def_property_readonly("get_place_by_operation_and_output_port",
                                  &ModelStat::get_place_by_operation_and_output_port);

    mdlStat.def_property_readonly("set_name_for_tensor", &ModelStat::set_name_for_tensor);
    mdlStat.def_property_readonly("add_name_for_tensor", &ModelStat::add_name_for_tensor);
    mdlStat.def_property_readonly("set_name_for_operation", &ModelStat::set_name_for_operation);
    mdlStat.def_property_readonly("free_name_for_tensor", &ModelStat::free_name_for_tensor);
    mdlStat.def_property_readonly("free_name_for_operation", &ModelStat::free_name_for_operation);
    mdlStat.def_property_readonly("set_name_for_dimension", &ModelStat::set_name_for_dimension);
    mdlStat.def_property_readonly("cut_and_add_new_input", &ModelStat::cut_and_add_new_input);
    mdlStat.def_property_readonly("cut_and_add_new_output", &ModelStat::cut_and_add_new_output);
    mdlStat.def_property_readonly("add_output", &ModelStat::add_output);
    mdlStat.def_property_readonly("remove_output", &ModelStat::remove_output);
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

static void register_mock_place_stat(py::module m) {
    m.def("get_place_stat", &PlaceMockPy::get_stat);
    m.def("clear_place_stat", &PlaceMockPy::clear_stat);

    py::class_<PlaceStat> placeStat(m, "PlaceStat", py::dynamic_attr());

    placeStat.def_property_readonly("lastArgString", &PlaceStat::get_lastArgString);
    placeStat.def_property_readonly("lastArgInt", &PlaceStat::get_lastArgInt);
    placeStat.def_property_readonly("lastArgPlace", &PlaceStat::get_lastArgPlace);

    placeStat.def_property_readonly("get_names", &PlaceStat::get_names);
    placeStat.def_property_readonly("get_consuming_operations", &PlaceStat::get_consuming_operations);
    placeStat.def_property_readonly("get_target_tensor", &PlaceStat::get_target_tensor);
    placeStat.def_property_readonly("get_producing_operation", &PlaceStat::get_producing_operation);
    placeStat.def_property_readonly("get_producing_port", &PlaceStat::get_producing_port);
    placeStat.def_property_readonly("get_input_port", &PlaceStat::get_input_port);
    placeStat.def_property_readonly("get_output_port", &PlaceStat::get_output_port);
    placeStat.def_property_readonly("get_consuming_ports", &PlaceStat::get_consuming_ports);
    placeStat.def_property_readonly("is_input", &PlaceStat::is_input);
    placeStat.def_property_readonly("is_output", &PlaceStat::is_output);
    placeStat.def_property_readonly("is_equal", &PlaceStat::is_equal);
    placeStat.def_property_readonly("is_equal_data", &PlaceStat::is_equal_data);
    placeStat.def_property_readonly("get_source_tensor", &PlaceStat::get_source_tensor);
}

static void register_frontend_wrappers(py::module m) {
#ifdef ENABLE_OV_PADDLE_FRONTEND
    py::class_<FrontEndWrapperPaddle, std::shared_ptr<FrontEndWrapperPaddle>> fe_paddle(m,
                                                                                        "FrontEndWrapperPaddle",
                                                                                        py::dynamic_attr());
    fe_paddle.def(py::init([]() {
        return std::make_shared<FrontEndWrapperPaddle>();
    }));
    fe_paddle.def(
        "add_extension",
        static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Extension>& extension)>(&FrontEnd::add_extension));
    fe_paddle.def("check_conversion_extension_registered", [](FrontEndWrapperPaddle& self, const std::string& name) {
        return self.check_conversion_extension_registered(name);
    });
#endif

#ifdef ENABLE_OV_TF_FRONTEND
    py::class_<FrontEndWrapperTensorflow, std::shared_ptr<FrontEndWrapperTensorflow>> fe_tensorflow(
        m,
        "FrontEndWrapperTensorflow",
        py::dynamic_attr());
    fe_tensorflow.def(py::init([]() {
        return std::make_shared<FrontEndWrapperTensorflow>();
    }));
    fe_tensorflow.def(
        "add_extension",
        static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Extension>& extension)>(&FrontEnd::add_extension));
    fe_tensorflow.def("add_extension",
                      static_cast<void (FrontEnd::*)(const std::vector<std::shared_ptr<ov::Extension>>& extension)>(
                          &FrontEnd::add_extension));
    fe_tensorflow.def("check_conversion_extension_registered",
                      [](FrontEndWrapperTensorflow& self, const std::string& name) {
                          return self.check_conversion_extension_registered(name);
                      });
#endif
}

PYBIND11_MODULE(pybind_mock_frontend, m) {
    m.doc() = "Mock frontend call counters for testing ov frontend bindings";
    register_mock_frontend_stat(m);
    register_mock_model_stat(m);
    register_mock_place_stat(m);
    register_frontend_wrappers(m);
}
