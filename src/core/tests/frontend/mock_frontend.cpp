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
        auto shape = Shape{2, 3, 300, 300};
        auto param = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        param->set_friendly_name("mock_param");
        param->set_layout("NCHW");
        std::vector<float> data(ov::shape_size(shape), 1.f);
        auto aligned_weights_buffer =
            std::make_shared<ngraph::runtime::AlignedBuffer>(shape_size(shape) * ::element::f32.size());
        auto weights = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
            aligned_weights_buffer->get_ptr<char>(),
            aligned_weights_buffer->size(),
            aligned_weights_buffer);
        auto constant = std::make_shared<ov::opset8::Constant>(ov::element::f32, shape, weights);
        constant->set_friendly_name("mock_const");
        auto op = std::make_shared<ov::opset8::Add>(param, constant);
        op->set_friendly_name("mock_add");
        auto op1 = std::make_shared<ov::opset8::Abs>(op);
        op1->set_friendly_name("mock_abs");
        auto split_axis = ov::opset8::Constant::create(ov::element::i32, {}, {0});
        split_axis->set_friendly_name("mock_split_axis");
        auto op2 = std::make_shared<ov::opset8::Split>(op1, split_axis, 2);
        op2->set_friendly_name("mock_split");
        auto res = std::make_shared<ov::opset8::Result>(op2->output(1));
        res->set_friendly_name("mock_result");
        auto param2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        param2->set_friendly_name("mock_param2");
        auto res2 = std::make_shared<ov::opset8::Result>(param2);
        res2->set_friendly_name("mock_result2");
        auto ov_model =
            std::make_shared<ov::Model>(ResultVector({res, res2}), ParameterVector({param, param2}), "mock1_model");
        ov_model->get_rt_info()["mock_test"] = std::string(1024, 't');
        ov_model->input(0).set_names({"mock_input"});
        ov_model->output(0).set_names({"mock_output"});
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
