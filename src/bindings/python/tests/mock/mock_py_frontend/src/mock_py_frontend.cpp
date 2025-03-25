// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_py_frontend/mock_py_frontend.hpp"

namespace ov {
namespace frontend {

FeStat FrontEndMockPy::m_stat = {};
ModelStat InputModelMockPy::m_stat = {};
PlaceStat PlaceMockPy::m_stat = {};

//--
std::vector<std::string> PlaceMockPy::get_names() const {
    m_stat.m_get_names++;
    return {};
}

std::vector<Place::Ptr> PlaceMockPy::get_consuming_operations() const {
    m_stat.m_get_consuming_operations++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = "";
    return {std::make_shared<PlaceMockPy>()};
}

std::vector<Place::Ptr> PlaceMockPy::get_consuming_operations(int outputPortIndex) const {
    m_stat.m_get_consuming_operations++;
    m_stat.m_lastArgInt = outputPortIndex;
    m_stat.m_lastArgString = "";
    return {std::make_shared<PlaceMockPy>()};
}

std::vector<Place::Ptr> PlaceMockPy::get_consuming_operations(const std::string& outputName) const {
    m_stat.m_get_consuming_operations++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = outputName;
    return {std::make_shared<PlaceMockPy>()};
}

std::vector<Place::Ptr> PlaceMockPy::get_consuming_operations(const std::string& outputName,
                                                              int outputPortIndex) const {
    m_stat.m_get_consuming_operations++;
    m_stat.m_lastArgInt = outputPortIndex;
    m_stat.m_lastArgString = outputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_target_tensor() const {
    m_stat.m_get_target_tensor++;
    m_stat.m_lastArgInt = -1;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_target_tensor(int outputPortIndex) const {
    m_stat.m_get_target_tensor++;
    m_stat.m_lastArgInt = outputPortIndex;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_target_tensor(const std::string& outputName) const {
    m_stat.m_get_target_tensor++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = outputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_target_tensor(const std::string& outputName, int outputPortIndex) const {
    m_stat.m_get_target_tensor++;
    m_stat.m_lastArgInt = outputPortIndex;
    m_stat.m_lastArgString = outputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_producing_operation() const {
    m_stat.m_get_producing_operation++;
    m_stat.m_lastArgInt = -1;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_producing_operation(int inputPortIndex) const {
    m_stat.m_get_producing_operation++;
    m_stat.m_lastArgInt = inputPortIndex;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_producing_operation(const std::string& inputName) const {
    m_stat.m_get_producing_operation++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = inputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    m_stat.m_get_producing_operation++;
    m_stat.m_lastArgInt = inputPortIndex;
    m_stat.m_lastArgString = inputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_producing_port() const {
    m_stat.m_get_producing_port++;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_input_port() const {
    m_stat.m_get_input_port++;
    m_stat.m_lastArgInt = -1;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_input_port(int inputPortIndex) const {
    m_stat.m_get_input_port++;
    m_stat.m_lastArgInt = inputPortIndex;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_input_port(const std::string& inputName) const {
    m_stat.m_get_input_port++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = inputName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_input_port(const std::string& inputName, int inputPortIndex) const {
    m_stat.m_get_input_port++;
    m_stat.m_lastArgInt = inputPortIndex;
    m_stat.m_lastArgString = inputName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_output_port() const {
    m_stat.m_get_output_port++;
    m_stat.m_lastArgInt = -1;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_output_port(int outputPortIndex) const {
    m_stat.m_get_output_port++;
    m_stat.m_lastArgInt = outputPortIndex;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_output_port(const std::string& outputName) const {
    m_stat.m_get_output_port++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = outputName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr PlaceMockPy::get_output_port(const std::string& outputName, int outputPortIndex) const {
    m_stat.m_get_output_port++;
    m_stat.m_lastArgInt = outputPortIndex;
    m_stat.m_lastArgString = outputName;
    return std::make_shared<PlaceMockPy>();
}

std::vector<Place::Ptr> PlaceMockPy::get_consuming_ports() const {
    m_stat.m_get_consuming_ports++;
    return {std::make_shared<PlaceMockPy>()};
}

bool PlaceMockPy::is_input() const {
    m_stat.m_is_input++;
    return false;
}

bool PlaceMockPy::is_output() const {
    m_stat.m_is_output++;
    return false;
}

bool PlaceMockPy::is_equal(const Ptr& another) const {
    m_stat.m_is_equal++;
    m_stat.m_lastArgPlace = another;
    return false;
}

bool PlaceMockPy::is_equal_data(const Ptr& another) const {
    m_stat.m_is_equal_data++;
    m_stat.m_lastArgPlace = another;
    return false;
}

Place::Ptr PlaceMockPy::get_source_tensor(int inputPortIndex) const {
    m_stat.m_get_source_tensor++;
    m_stat.m_lastArgInt = inputPortIndex;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_source_tensor() const {
    m_stat.m_get_source_tensor++;
    m_stat.m_lastArgInt = -1;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_source_tensor(const std::string& inputName) const {
    m_stat.m_get_source_tensor++;
    m_stat.m_lastArgInt = -1;
    m_stat.m_lastArgString = inputName;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr PlaceMockPy::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    m_stat.m_get_source_tensor++;
    m_stat.m_lastArgInt = inputPortIndex;
    m_stat.m_lastArgString = inputName;
    return {std::make_shared<PlaceMockPy>()};
}

PlaceStat PlaceMockPy::get_stat() {
    return m_stat;
}

void PlaceMockPy::clear_stat() {
    m_stat = {};
}

//--
std::vector<Place::Ptr> InputModelMockPy::get_inputs() const {
    m_stat.m_get_inputs++;
    return {std::make_shared<PlaceMockPy>()};
}

std::vector<Place::Ptr> InputModelMockPy::get_outputs() const {
    m_stat.m_get_outputs++;
    return {std::make_shared<PlaceMockPy>()};
}

Place::Ptr InputModelMockPy::get_place_by_tensor_name(const std::string& tensorName) const {
    m_stat.m_get_place_by_tensor_name++;
    m_stat.m_lastArgString = tensorName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr InputModelMockPy::get_place_by_operation_name(const std::string& operationName) const {
    m_stat.m_get_place_by_operation_name++;
    m_stat.m_lastArgString = operationName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr InputModelMockPy::get_place_by_operation_name_and_input_port(const std::string& operationName,
                                                                        int inputPortIndex) {
    m_stat.m_get_place_by_operation_and_input_port++;
    m_stat.m_lastArgInt = inputPortIndex;
    m_stat.m_lastArgString = operationName;
    return std::make_shared<PlaceMockPy>();
}

Place::Ptr InputModelMockPy::get_place_by_operation_name_and_output_port(const std::string& operationName,
                                                                         int outputPortIndex) {
    m_stat.m_get_place_by_operation_and_output_port++;
    m_stat.m_lastArgInt = outputPortIndex;
    m_stat.m_lastArgString = operationName;
    return std::make_shared<PlaceMockPy>();
}

void InputModelMockPy::set_name_for_tensor(const Place::Ptr& tensor, const std::string& newName) {
    m_stat.m_set_name_for_tensor++;
    m_stat.m_lastArgPlace = tensor;
    m_stat.m_lastArgString = newName;
}

void InputModelMockPy::add_name_for_tensor(const Place::Ptr& tensor, const std::string& newName) {
    m_stat.m_add_name_for_tensor++;
    m_stat.m_lastArgPlace = tensor;
    m_stat.m_lastArgString = newName;
}

void InputModelMockPy::set_name_for_operation(const Place::Ptr& operation, const std::string& newName) {
    m_stat.m_set_name_for_operation++;
    m_stat.m_lastArgPlace = operation;
    m_stat.m_lastArgString = newName;
}

void InputModelMockPy::free_name_for_tensor(const std::string& name) {
    m_stat.m_free_name_for_tensor++;
    m_stat.m_lastArgString = name;
}

void InputModelMockPy::free_name_for_operation(const std::string& name) {
    m_stat.m_free_name_for_operation++;
    m_stat.m_lastArgString = name;
}

void InputModelMockPy::set_name_for_dimension(const Place::Ptr& place,
                                              size_t shapeDimIndex,
                                              const std::string& dimName) {
    m_stat.m_set_name_for_dimension++;
    m_stat.m_lastArgPlace = place;
    m_stat.m_lastArgInt = static_cast<int>(shapeDimIndex);
    m_stat.m_lastArgString = dimName;
}

void InputModelMockPy::cut_and_add_new_input(const Place::Ptr& place, const std::string& newNameOptional) {
    m_stat.m_cut_and_add_new_input++;
    m_stat.m_lastArgPlace = place;
    m_stat.m_lastArgString = newNameOptional;
}

void InputModelMockPy::cut_and_add_new_output(const Place::Ptr& place, const std::string& newNameOptional) {
    m_stat.m_cut_and_add_new_output++;
    m_stat.m_lastArgPlace = place;
    m_stat.m_lastArgString = newNameOptional;
}

Place::Ptr InputModelMockPy::add_output(const Place::Ptr& place) {
    m_stat.m_add_output++;
    m_stat.m_lastArgPlace = place;
    return std::make_shared<PlaceMockPy>();
}

void InputModelMockPy::remove_output(const Place::Ptr& place) {
    m_stat.m_remove_output++;
    m_stat.m_lastArgPlace = place;
}

void InputModelMockPy::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    m_stat.m_override_all_outputs++;
    m_stat.m_lastArgOutputPlaces = outputs;
}

void InputModelMockPy::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    m_stat.m_override_all_inputs++;
    m_stat.m_lastArgInputPlaces = inputs;
}

void InputModelMockPy::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    m_stat.m_extract_subgraph++;
    m_stat.m_lastArgInputPlaces = inputs;
    m_stat.m_lastArgOutputPlaces = outputs;
}

// Setting tensor properties
void InputModelMockPy::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    m_stat.m_set_partial_shape++;
    m_stat.m_lastArgPlace = place;
    m_stat.m_lastArgPartialShape = shape;
}

PartialShape InputModelMockPy::get_partial_shape(const Place::Ptr& place) const {
    m_stat.m_get_partial_shape++;
    m_stat.m_lastArgPlace = place;
    return {};
}

void InputModelMockPy::set_element_type(const Place::Ptr& place, const element::Type& type) {
    m_stat.m_set_element_type++;
    m_stat.m_lastArgPlace = place;
    m_stat.m_lastArgElementType = type;
}

ModelStat InputModelMockPy::get_stat() {
    return m_stat;
}

void InputModelMockPy::clear_stat() {
    m_stat = {};
}

//--
InputModel::Ptr FrontEndMockPy::load_impl(const std::vector<ov::Any>& params) const {
    if (m_telemetry) {
        m_telemetry->send_event("load_impl", "label", 42);
        m_telemetry->send_error("load_impl_error");
        m_telemetry->send_stack_trace("mock_stack_trace");
    }
    if (!params.empty()) {
        OPENVINO_ASSERT(params[0].is<std::string>(), "Only path is supported.");
        m_stat.m_load_paths.push_back(params[0].as<std::string>());
    }

    return std::make_shared<InputModelMockPy>();
}

bool FrontEndMockPy::supported_impl(const std::vector<ov::Any>& params) const {
    m_stat.m_supported++;
    if (!params.empty() && params[0].is<std::string>()) {
        auto path = params[0].as<std::string>();
        if (path.find(".test_mock_py_mdl") != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<ov::Model> FrontEndMockPy::convert(const InputModel::Ptr& model) const {
    m_stat.m_convert_model++;
    return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
}

void FrontEndMockPy::convert(const std::shared_ptr<ov::Model>& func) const {
    m_stat.m_convert++;
}

std::shared_ptr<ov::Model> FrontEndMockPy::convert_partially(const InputModel::Ptr& model) const {
    m_stat.m_convert_partially++;
    return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
}

std::shared_ptr<ov::Model> FrontEndMockPy::decode(const InputModel::Ptr& model) const {
    m_stat.m_decode++;
    return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
}

void FrontEndMockPy::normalize(const std::shared_ptr<ov::Model>& function) const {
    m_stat.m_normalize++;
}

std::string FrontEndMockPy::get_name() const {
    m_stat.m_get_name++;
    return "mock_py";
}

void FrontEndMockPy::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto p = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = p;
    }
}

FeStat FrontEndMockPy::get_stat() {
    return m_stat;
}

void FrontEndMockPy::clear_stat() {
    m_stat = {};
}
}  // namespace frontend
}  // namespace ov

MOCK_C_API ov::frontend::FrontEndVersion get_api_version();
MOCK_C_API void* get_front_end_data();

MOCK_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

MOCK_C_API void* get_front_end_data() {
    ov::frontend::FrontEndPluginInfo* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "mock_py";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::FrontEndMockPy>();
    };
    return res;
}
