// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/manager.hpp"
#include "visibility.hpp"

// OK to have 'using' in mock header
using namespace ngraph;
using namespace ov::frontend;

////////////////////////////////

struct MOCK_API PlaceStat {
    int m_get_names = 0;
    int m_get_consuming_operations = 0;
    int m_get_target_tensor = 0;
    int m_get_producing_operation = 0;
    int m_get_producing_port = 0;
    int m_get_input_port = 0;
    int m_get_output_port = 0;
    int m_get_consuming_ports = 0;
    int m_is_input = 0;
    int m_is_output = 0;
    int m_is_equal = 0;
    int m_is_equal_data = 0;
    int m_get_source_tensor = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt;
    Place::Ptr m_lastArgPlace = nullptr;

    // Getters
    int get_names() const {
        return m_get_names;
    }
    int get_consuming_operations() const {
        return m_get_consuming_operations;
    }
    int get_target_tensor() const {
        return m_get_target_tensor;
    }
    int get_producing_operation() const {
        return m_get_producing_operation;
    }
    int get_producing_port() const {
        return m_get_producing_port;
    }
    int get_input_port() const {
        return m_get_input_port;
    }
    int get_output_port() const {
        return m_get_output_port;
    }
    int get_consuming_ports() const {
        return m_get_consuming_ports;
    }
    int is_input() const {
        return m_is_input;
    }
    int is_output() const {
        return m_is_output;
    }
    int is_equal() const {
        return m_is_equal;
    }
    int is_equal_data() const {
        return m_is_equal_data;
    }
    int get_source_tensor() const {
        return m_get_source_tensor;
    }

    // Arguments getters
    std::string get_lastArgString() const {
        return m_lastArgString;
    }
    int get_lastArgInt() const {
        return m_lastArgInt;
    }
    Place::Ptr get_lastArgPlace() const {
        return m_lastArgPlace;
    }
};

class MOCK_API PlaceMockPy : public Place {
    static PlaceStat m_stat;

public:
    std::vector<std::string> get_names() const override {
        m_stat.m_get_names++;
        return {};
    }

    std::vector<Place::Ptr> get_consuming_operations() const override {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = "";
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_consuming_operations(int outputPortIndex) const override {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = "";
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_consuming_operations(const std::string& outputName) const override {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = outputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_consuming_operations(const std::string& outputName,
                                                     int outputPortIndex) const override {
        m_stat.m_get_consuming_operations++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = outputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_target_tensor() const override {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_target_tensor(int outputPortIndex) const override {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = outputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_target_tensor(const std::string& outputName) const override {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = outputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_target_tensor(const std::string& outputName, int outputPortIndex) const override {
        m_stat.m_get_target_tensor++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = outputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_producing_operation() const override {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_producing_operation(int inputPortIndex) const override {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = inputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_producing_operation(const std::string& inputName) const override {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = inputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_producing_operation(const std::string& inputName, int inputPortIndex) const override {
        m_stat.m_get_producing_operation++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = inputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_producing_port() const override {
        m_stat.m_get_producing_port++;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port() const override {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(int inputPortIndex) const override {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(const std::string& inputName) const override {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = inputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override {
        m_stat.m_get_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = inputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port() const override {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = -1;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(int outputPortIndex) const override {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(const std::string& outputName) const override {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = outputName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_output_port(const std::string& outputName, int outputPortIndex) const override {
        m_stat.m_get_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = outputName;
        return std::make_shared<PlaceMockPy>();
    }

    std::vector<Place::Ptr> get_consuming_ports() const override {
        m_stat.m_get_consuming_ports++;
        return {std::make_shared<PlaceMockPy>()};
    }

    bool is_input() const override {
        m_stat.m_is_input++;
        return false;
    }

    bool is_output() const override {
        m_stat.m_is_output++;
        return false;
    }

    bool is_equal(const Ptr& another) const override {
        m_stat.m_is_equal++;
        m_stat.m_lastArgPlace = another;
        return false;
    }

    bool is_equal_data(const Ptr& another) const override {
        m_stat.m_is_equal_data++;
        m_stat.m_lastArgPlace = another;
        return false;
    }

    Place::Ptr get_source_tensor(int inputPortIndex) const override {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = inputPortIndex;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_source_tensor() const override {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = -1;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_source_tensor(const std::string& inputName) const override {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = -1;
        m_stat.m_lastArgString = inputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_source_tensor(const std::string& inputName, int inputPortIndex) const override {
        m_stat.m_get_source_tensor++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = inputName;
        return {std::make_shared<PlaceMockPy>()};
    }

    //---------------Stat--------------------
    static PlaceStat get_stat() {
        return m_stat;
    }

    static void clear_stat() {
        m_stat = {};
    }
};

////////////////////////////////

struct MOCK_API ModelStat {
    int m_get_inputs = 0;
    int m_get_outputs = 0;
    int m_get_place_by_tensor_name = 0;
    int m_get_place_by_operation_name = 0;
    int m_get_place_by_operation_and_input_port = 0;
    int m_get_place_by_operation_and_output_port = 0;
    int m_set_name_for_tensor = 0;
    int m_add_name_for_tensor = 0;
    int m_set_name_for_operation = 0;
    int m_free_name_for_tensor = 0;
    int m_free_name_for_operation = 0;
    int m_set_name_for_dimension = 0;
    int m_cut_and_add_new_input = 0;
    int m_cut_and_add_new_output = 0;
    int m_add_output = 0;
    int m_remove_output = 0;
    int m_set_partial_shape = 0;
    int m_get_partial_shape = 0;
    int m_set_element_type = 0;

    int m_extract_subgraph = 0;
    int m_override_all_inputs = 0;
    int m_override_all_outputs = 0;

    // Arguments tracking
    std::string m_lastArgString;
    int m_lastArgInt = -1;
    Place::Ptr m_lastArgPlace = nullptr;
    std::vector<Place::Ptr> m_lastArgInputPlaces;
    std::vector<Place::Ptr> m_lastArgOutputPlaces;
    ngraph::element::Type m_lastArgElementType;
    ngraph::PartialShape m_lastArgPartialShape;

    // Getters
    int get_inputs() const {
        return m_get_inputs;
    }
    int get_outputs() const {
        return m_get_outputs;
    }
    int extract_subgraph() const {
        return m_extract_subgraph;
    }
    int override_all_inputs() const {
        return m_override_all_inputs;
    }
    int override_all_outputs() const {
        return m_override_all_outputs;
    }
    int get_place_by_tensor_name() const {
        return m_get_place_by_tensor_name;
    }
    int get_place_by_operation_name() const {
        return m_get_place_by_operation_name;
    }
    int get_place_by_operation_and_input_port() const {
        return m_get_place_by_operation_and_input_port;
    }
    int get_place_by_operation_and_output_port() const {
        return m_get_place_by_operation_and_output_port;
    }
    int set_name_for_tensor() const {
        return m_set_name_for_tensor;
    }
    int add_name_for_tensor() const {
        return m_add_name_for_tensor;
    }
    int set_name_for_operation() const {
        return m_set_name_for_operation;
    }
    int free_name_for_tensor() const {
        return m_free_name_for_tensor;
    }
    int free_name_for_operation() const {
        return m_free_name_for_operation;
    }
    int set_name_for_dimension() const {
        return m_set_name_for_dimension;
    }
    int cut_and_add_new_input() const {
        return m_cut_and_add_new_input;
    }
    int cut_and_add_new_output() const {
        return m_cut_and_add_new_output;
    }
    int add_output() const {
        return m_add_output;
    }
    int remove_output() const {
        return m_remove_output;
    }
    int set_partial_shape() const {
        return m_set_partial_shape;
    }
    int get_partial_shape() const {
        return m_get_partial_shape;
    }
    int set_element_type() const {
        return m_set_element_type;
    }

    // Arguments getters
    std::string get_lastArgString() const {
        return m_lastArgString;
    }
    int get_lastArgInt() const {
        return m_lastArgInt;
    }
    Place::Ptr get_lastArgPlace() const {
        return m_lastArgPlace;
    }
    std::vector<Place::Ptr> get_lastArgInputPlaces() const {
        return m_lastArgInputPlaces;
    }
    std::vector<Place::Ptr> get_lastArgOutputPlaces() const {
        return m_lastArgOutputPlaces;
    }
    ngraph::element::Type get_lastArgElementType() const {
        return m_lastArgElementType;
    }
    ngraph::PartialShape get_lastArgPartialShape() const {
        return m_lastArgPartialShape;
    }
};

class MOCK_API InputModelMockPy : public InputModel {
    static ModelStat m_stat;

public:
    std::vector<Place::Ptr> get_inputs() const override {
        m_stat.m_get_inputs++;
        return {std::make_shared<PlaceMockPy>()};
    }

    std::vector<Place::Ptr> get_outputs() const override {
        m_stat.m_get_outputs++;
        return {std::make_shared<PlaceMockPy>()};
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override {
        m_stat.m_get_place_by_tensor_name++;
        m_stat.m_lastArgString = tensorName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name(const std::string& operationName) const override {
        m_stat.m_get_place_by_operation_name++;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operationName,
                                                          int inputPortIndex) override {
        m_stat.m_get_place_by_operation_and_input_port++;
        m_stat.m_lastArgInt = inputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    Place::Ptr get_place_by_operation_name_and_output_port(const std::string& operationName,
                                                           int outputPortIndex) override {
        m_stat.m_get_place_by_operation_and_output_port++;
        m_stat.m_lastArgInt = outputPortIndex;
        m_stat.m_lastArgString = operationName;
        return std::make_shared<PlaceMockPy>();
    }

    void set_name_for_tensor(const Place::Ptr& tensor, const std::string& newName) override {
        m_stat.m_set_name_for_tensor++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void add_name_for_tensor(const Place::Ptr& tensor, const std::string& newName) override {
        m_stat.m_add_name_for_tensor++;
        m_stat.m_lastArgPlace = tensor;
        m_stat.m_lastArgString = newName;
    }

    void set_name_for_operation(const Place::Ptr& operation, const std::string& newName) override {
        m_stat.m_set_name_for_operation++;
        m_stat.m_lastArgPlace = operation;
        m_stat.m_lastArgString = newName;
    }

    void free_name_for_tensor(const std::string& name) override {
        m_stat.m_free_name_for_tensor++;
        m_stat.m_lastArgString = name;
    }

    void free_name_for_operation(const std::string& name) override {
        m_stat.m_free_name_for_operation++;
        m_stat.m_lastArgString = name;
    }

    void set_name_for_dimension(const Place::Ptr& place, size_t shapeDimIndex, const std::string& dimName) override {
        m_stat.m_set_name_for_dimension++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgInt = static_cast<int>(shapeDimIndex);
        m_stat.m_lastArgString = dimName;
    }

    void cut_and_add_new_input(const Place::Ptr& place, const std::string& newNameOptional) override {
        m_stat.m_cut_and_add_new_input++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    void cut_and_add_new_output(const Place::Ptr& place, const std::string& newNameOptional) override {
        m_stat.m_cut_and_add_new_output++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgString = newNameOptional;
    }

    Place::Ptr add_output(const Place::Ptr& place) override {
        m_stat.m_add_output++;
        m_stat.m_lastArgPlace = place;
        return std::make_shared<PlaceMockPy>();
    }

    void remove_output(const Place::Ptr& place) override {
        m_stat.m_remove_output++;
        m_stat.m_lastArgPlace = place;
    }

    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override {
        m_stat.m_override_all_outputs++;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override {
        m_stat.m_override_all_inputs++;
        m_stat.m_lastArgInputPlaces = inputs;
    }

    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override {
        m_stat.m_extract_subgraph++;
        m_stat.m_lastArgInputPlaces = inputs;
        m_stat.m_lastArgOutputPlaces = outputs;
    }

    // Setting tensor properties
    void set_partial_shape(const Place::Ptr& place, const ngraph::PartialShape& shape) override {
        m_stat.m_set_partial_shape++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgPartialShape = shape;
    }

    ngraph::PartialShape get_partial_shape(const Place::Ptr& place) const override {
        m_stat.m_get_partial_shape++;
        m_stat.m_lastArgPlace = place;
        return {};
    }

    void set_element_type(const Place::Ptr& place, const ngraph::element::Type& type) override {
        m_stat.m_set_element_type++;
        m_stat.m_lastArgPlace = place;
        m_stat.m_lastArgElementType = type;
    }

    //---------------Stat--------------------
    static ModelStat get_stat() {
        return m_stat;
    }

    static void clear_stat() {
        m_stat = {};
    }
};

/////////////////////////////////////////////////////////

struct MOCK_API FeStat {
    std::vector<std::string> m_load_paths;
    int m_convert_model = 0;
    int m_convert = 0;
    int m_convert_partially = 0;
    int m_decode = 0;
    int m_normalize = 0;
    int m_get_name = 0;
    int m_supported = 0;
    // Getters
    std::vector<std::string> load_paths() const {
        return m_load_paths;
    }
    int convert_model() const {
        return m_convert_model;
    }
    int convert() const {
        return m_convert;
    }
    int convert_partially() const {
        return m_convert_partially;
    }
    int decode() const {
        return m_decode;
    }
    int normalize() const {
        return m_normalize;
    }
    int get_name() const {
        return m_get_name;
    }

    int supported() const {
        return m_supported;
    }
};

class MOCK_API FrontEndMockPy : public FrontEnd {
    static FeStat m_stat;
    std::shared_ptr<ov::frontend::TelemetryExtension> m_telemetry;

public:
    FrontEndMockPy() = default;

    InputModel::Ptr load_impl(const std::vector<ov::Any>& params) const override {
        if (m_telemetry) {
            m_telemetry->send_event("load_impl", "label", 42);
            m_telemetry->send_error("load_impl_error");
            m_telemetry->send_stack_trace("mock_stack_trace");
        }
        if (!params.empty() && params[0].is<std::string>())
            m_stat.m_load_paths.push_back(params[0].as<std::string>());
        return std::make_shared<InputModelMockPy>();
    }

    bool supported_impl(const std::vector<ov::Any>& params) const override {
        m_stat.m_supported++;
        if (!params.empty() && params[0].is<std::string>()) {
            auto path = params[0].as<std::string>();
            if (path.find(".test_mock_py_mdl") != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override {
        m_stat.m_convert_model++;
        return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
    }

    void convert(const std::shared_ptr<ov::Model>& func) const override {
        m_stat.m_convert++;
    }

    std::shared_ptr<ov::Model> convert_partially(const InputModel::Ptr& model) const override {
        m_stat.m_convert_partially++;
        return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
    }

    std::shared_ptr<ov::Model> decode(const InputModel::Ptr& model) const override {
        m_stat.m_decode++;
        return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
    }

    void normalize(const std::shared_ptr<ov::Model>& function) const override {
        m_stat.m_normalize++;
    }

    std::string get_name() const override {
        m_stat.m_get_name++;
        return "mock_py";
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        if (auto p = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
            m_telemetry = p;
        }
    }

    static FeStat get_stat() {
        return m_stat;
    }

    static void clear_stat() {
        m_stat = {};
    }
};
