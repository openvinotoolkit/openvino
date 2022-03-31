// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/opsets/opset8.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef openvino_mock1_frontend_EXPORTS
#    define MOCK_API OPENVINO_CORE_EXPORTS
#else
#    define MOCK_API OPENVINO_CORE_IMPORTS
#endif  // openvino_mock1_frontend_EXPORTS

using namespace ngraph;
using namespace ov::frontend;

class InputModelMock : public InputModel {
public:
    bool m_throw = false;

    std::vector<Place::Ptr> get_inputs() const override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    std::vector<Place::Ptr> get_outputs() const override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    Place::Ptr get_place_by_operation_name(const std::string& operation_name) const override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                          int input_port_index) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    Place::Ptr get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                           int output_port_index) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    void set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void free_name_for_tensor(const std::string& name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void free_name_for_operation(const std::string& name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    Place::Ptr add_output(const Place::Ptr& place) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    void remove_output(const Place::Ptr& place) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void set_partial_shape(const Place::Ptr& place, const PartialShape& shape) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    PartialShape get_partial_shape(const Place::Ptr& place) const override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
        return {};
    }

    void set_element_type(const Place::Ptr& place, const element::Type& type) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void set_tensor_value(const Place::Ptr& place, const void* value) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }

    void set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value) override {
        FRONT_END_GENERAL_CHECK(!m_throw, "Test exception");
    }
};

class FrontEndMock : public FrontEnd {
    mutable bool m_throw_next{false};

public:
    std::string get_name() const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
        return "mock1";
    }

    bool supported_impl(const std::vector<ov::Any>& variants) const override {
        if (variants.size() == 1 && variants[0].is<std::string>()) {
            std::string command = variants[0].as<std::string>();
            FRONT_END_GENERAL_CHECK(command != "throw_now", "Test exception");
        }
        return false;
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
    }

    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override {
        auto input_model = std::make_shared<InputModelMock>();
        if (variants.size() == 1 && variants[0].is<std::string>()) {
            std::string command = variants[0].as<std::string>();
            if (command == "throw_now") {
                OPENVINO_UNREACHABLE("Test throw load input model");
            } else if (command == "throw_next") {
                m_throw_next = true;
            } else if (command == "throw_model") {
                input_model->m_throw = true;
            }
        }
        return input_model;
    }

    std::shared_ptr<ov::Model> convert_partially(const InputModel::Ptr& model) const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
        return nullptr;
    }

    std::shared_ptr<ov::Model> decode(const InputModel::Ptr& model) const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");

        return nullptr;
    }

    void convert(const std::shared_ptr<ov::Model>& model) const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
    }

    void normalize(const std::shared_ptr<ov::Model>& model) const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
    }

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override {
        FRONT_END_GENERAL_CHECK(!m_throw_next, "Test exception");
        auto shape = Shape{1, 2, 300, 300};
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        std::vector<float> data(ov::shape_size(shape), 1.f);
        auto constant = ov::opset8::Constant::create(ov::element::f32, shape, data);
        auto op = std::make_shared<ov::opset8::Add>(param, constant);
        auto res = std::make_shared<ov::opset8::Result>(op);
        auto ov_model = std::make_shared<ov::Model>(ResultVector({res}), ParameterVector({param}), "mock1_model");
        ov_model->get_rt_info()["mock_test"] = std::string(1024, 't');
        return ov_model;
    }
};

extern "C" MOCK_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" MOCK_API void* GetFrontEndData() {
    auto* res = new FrontEndPluginInfo();
    res->m_name = "mock1";
    res->m_creator = []() {
        return std::make_shared<FrontEndMock>();
    };
    return res;
}
