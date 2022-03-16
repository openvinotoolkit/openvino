// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/evaluate_extension.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/runtime.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "so_extension.hpp"

inline std::string get_extension_path() {
    return ov::util::make_plugin_library_name<char>({}, std::string("openvino_template_extension") + IE_BUILD_POSTFIX);
}

TEST(extension, load_extension) {
    EXPECT_NO_THROW(ov::detail::load_extensions(get_extension_path()));
}

TEST(extension, load_extension_and_cast) {
    std::vector<ov::Extension::Ptr> so_extensions = ov::detail::load_extensions(get_extension_path());
    ASSERT_EQ(1, so_extensions.size());
    std::vector<ov::Extension::Ptr> extensions;
    std::vector<std::shared_ptr<void>> so;
    for (const auto& ext : so_extensions) {
        if (auto so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext)) {
            extensions.emplace_back(so_ext->extension());
            so.emplace_back(so_ext->shared_object());
        }
    }
    so_extensions.clear();
    EXPECT_EQ(1, extensions.size());
    EXPECT_NE(nullptr, dynamic_cast<ov::BaseOpExtension*>(extensions[0].get()));
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<ov::BaseOpExtension>(extensions[0]));
    extensions.clear();
}

class Add1Evaluate : public ov::EvaluateExtension {
public:
    const ov::DiscreteTypeInfo& get_type_info() const override {
        return ov::op::v0::Relu::get_type_info_static();
    }
    bool support_evaluate(const std::shared_ptr<const ov::Node>& node,
                          const std::vector<std::type_info>& input_tensor_types = {},
                          const std::vector<std::type_info>& output_tensor_types = {}) const override {
        if (node->get_type_info() != ov::op::v0::Relu::get_type_info_static() ||
            node->get_input_element_type(0) != ov::element::i64) {
            return false;
        }
        CHECK_TENSOR_TYPES(node, input_tensor_types, output_tensor_types, ov::Tensor);
        return true;
    }
    bool evaluate(const std::shared_ptr<const ov::Node>& node,
                  ov::TensorVector& output_values,
                  const ov::TensorVector& input_values) const override {
        if (input_values[0].get_element_type() != ov::element::i64 || !is_host_tensors(input_values) ||
            !is_host_tensors(output_values))
            return false;
        auto input_tensor = input_values[0];
        auto output_tensor = output_values[0];

        const auto* input_data = input_tensor.data<int64_t>();
        auto* output_data = output_tensor.data<int64_t>();
        for (size_t i = 0; i < input_tensor.get_size(); i++) {
            output_data[i] = input_data[i] + 1;
        }
        return true;
    }
};

class Add1EvaluateRemote : public ov::EvaluateExtension {
public:
    const ov::DiscreteTypeInfo& get_type_info() const override {
        return ov::op::v0::Relu::get_type_info_static();
    }
    bool support_evaluate(const std::shared_ptr<const ov::Node>& node,
                          const std::vector<std::type_info>& input_tensor_types = {},
                          const std::vector<std::type_info>& output_tensor_types = {}) const override {
        if (node->get_type_info() != ov::op::v0::Relu::get_type_info_static()) {
            return false;
        }
        CHECK_TENSOR_TYPES(node, input_tensor_types, output_tensor_types, ov::RemoteTensor);
        return true;
    }
    bool evaluate(const std::shared_ptr<const ov::Node>& node,
                  ov::TensorVector& output_values,
                  const ov::TensorVector& input_values) const override {
        if (is_host_tensors(input_values) || is_host_tensors(output_values)) {
            std::cout << "REMOTE EVALUATE false" << std::endl;
            return false;
        }
        OPENVINO_ASSERT(false, "Evaluate was called for remote tensors!");
    }
};

namespace {

std::string& get_dev_name() {
    static std::string dev_name = "TEMPLATE";
    return dev_name;
}

template <class T>
class OpModelEvaluate : public ov::EvaluateExtension {
public:
    const ov::DiscreteTypeInfo& get_type_info() const override {
        return T::get_type_info_static();
    }
    bool support_evaluate(const std::shared_ptr<const ov::Node>& node,
                          const std::vector<std::type_info>& input_tensor_types = {},
                          const std::vector<std::type_info>& output_tensor_types = {}) const override {
        if (node->get_type_info() != T::get_type_info_static()) {
            return false;
        }
        CHECK_TENSOR_TYPES(node, input_tensor_types, output_tensor_types, ov::Tensor);
        return true;
    }
    bool evaluate(const std::shared_ptr<const ov::Node>& node,
                  ov::TensorVector& output_values,
                  const ov::TensorVector& input_values) const override {
        if (!is_host_tensors(input_values) || !is_host_tensors(output_values))
            return false;
        std::cout << "EVALUATE" << std::endl;
        std::shared_ptr<ov::Model> model;
        // Create single operation model
        {
            ov::ParameterVector params;
            ov::OutputVector inputs;
            for (const auto& tensor : input_values) {
                auto param = std::make_shared<ov::opset8::Parameter>(tensor.get_element_type(), tensor.get_shape());
                params.emplace_back(param);
                inputs.emplace_back(param);
            }
            auto cloned_node = node->clone_with_new_inputs(inputs);
            model = std::make_shared<ov::Model>(cloned_node->outputs(), params);
        }

        ov::Core core;
        try {
            // Register template plugin
            if (get_dev_name() == "TEMPLATE")
                core.register_plugin(std::string("openvino_template_plugin") + IE_BUILD_POSTFIX, "TEMPLATE");
        } catch (...) {
        }
        auto infer_request = core.compile_model(model, get_dev_name()).create_infer_request();
        for (size_t i = 0; i < input_values.size(); i++) {
            infer_request.set_input_tensor(i, input_values[i]);
        }
        for (size_t i = 0; i < output_values.size(); i++) {
            infer_request.set_output_tensor(i, output_values[i]);
        }
        infer_request.infer();
        return true;
    }
};

template <class OP, class Ext>
class TestExtension {
public:
    TestExtension() {
        OP::add_extension(std::make_shared<Ext>());
    }
    ~TestExtension() {
        ov::get_extensions_for_type(OP::get_type_info_static()).clear();
    }
};

template <class T, class Ext = Add1Evaluate>
class TestReluEvaluate {
public:
    TestReluEvaluate(const ov::element::Type& el_type,
                     const ov::Shape& shape,
                     const std::vector<T>& in_data,
                     const std::vector<T>& out_data,
                     bool add_extension = false) {
        if (add_extension)
            ext = std::make_shared<TestExtension<ov::opset8::Relu, Ext>>();

        auto parameter = std::make_shared<ov::opset8::Parameter>(el_type, shape);
        std::shared_ptr<ov::Node> relu = std::make_shared<ov::opset8::Relu>(parameter);
        const void* in_data_ptr = in_data.data();
        ov::Tensor input_tensor(el_type, shape, const_cast<void*>(in_data_ptr));
        ov::Tensor output_tensor(el_type, shape);
        ov::TensorVector output_tensors = {output_tensor};
        ov::TensorVector input_tensors = {input_tensor};
        relu->evaluate(output_tensors, input_tensors);
        m_success = std::memcmp(out_data.data(), output_tensor.data(), output_tensor.get_byte_size()) == 0;
    }

    bool success() const {
        return m_success;
    }

private:
    bool m_success = false;
    std::shared_ptr<void> ext;
};

}  // namespace

TEST(extension, evaluate_extension_relu) {
    std::vector<int64_t> orig_data = {-2, -1, 1, 2};
    std::vector<int64_t> ref_orig_data = {0, 0, 1, 2};
    std::vector<int64_t> ref_data = {-1, 0, 2, 3};
    ov::element::Type el_type = ov::element::i64;
    ov::Shape shape({1, 1, 2, 2});
    {
        TestReluEvaluate<int64_t> test_eval(el_type, shape, orig_data, ref_data, true);
        EXPECT_TRUE(test_eval.success());
    }

    {
        TestReluEvaluate<int64_t> test_eval(el_type, shape, orig_data, ref_orig_data, false);
        EXPECT_TRUE(test_eval.success());
    }
}

TEST(extension, evaluate_extension_relu_i32) {
    std::vector<int32_t> orig_data = {-2, -1, 1, 2};
    std::vector<int32_t> ref_orig_data = {0, 0, 1, 2};
    std::vector<int32_t> ref_data = {-1, 0, 2, 3};
    ov::element::Type el_type = ov::element::i32;
    ov::Shape shape({1, 1, 2, 2});
    {
        TestReluEvaluate<int32_t> test_eval(el_type, shape, orig_data, ref_data, true);
        EXPECT_FALSE(test_eval.success());
    }

    {
        TestReluEvaluate<int32_t> test_eval(el_type, shape, orig_data, ref_orig_data, false);
        EXPECT_TRUE(test_eval.success());
    }
}

TEST(extension, evaluate_extension_relu_remote) {
    std::vector<int64_t> orig_data = {-2, -1, 1, 2};
    std::vector<int64_t> ref_orig_data = {0, 0, 1, 2};
    std::vector<int64_t> ref_data = {-1, 0, 2, 3};
    ov::element::Type el_type = ov::element::i32;
    ov::Shape shape({1, 1, 2, 2});
    {
        TestReluEvaluate<int64_t, Add1EvaluateRemote> test_eval(el_type, shape, orig_data, ref_data, true);
        EXPECT_FALSE(test_eval.success());
    }

    {
        TestReluEvaluate<int64_t, Add1EvaluateRemote> test_eval(el_type, shape, orig_data, ref_orig_data, false);
        EXPECT_TRUE(test_eval.success());
    }
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

inline double get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
};

TEST(extension, constant_fold_conv) {
    const auto& gen_data = [](const ov::Shape& shape) {
        size_t size = ov::shape_size(shape);
        std::vector<float> data(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
        return data;
    };
    std::shared_ptr<ov::Model> origin_model;
    {
        ov::Shape data_shape{1, 7, 320, 32, 32};
        ov::Shape weights_shape{32, 7, 3, 3, 3};
        ov::Shape bias_shape{1, 32, 1, 1, 1};
        std::vector<float> in_data = gen_data(data_shape);
        std::vector<float> w_data = gen_data(weights_shape);
        std::vector<float> b_data = gen_data(bias_shape);
        auto const_data = ov::opset8::Constant::create(ov::element::f32, data_shape, in_data);
        auto weights = ov::opset8::Constant::create(ov::element::f32, weights_shape, w_data);
        auto conv = std::make_shared<ov::opset8::Convolution>(const_data,
                                                              weights,
                                                              ov::Strides{3, 3, 3},
                                                              ov::CoordinateDiff({0, 0, 0}),
                                                              ov::CoordinateDiff({0, 0, 0}),
                                                              ov::Strides{2, 2, 2});
        auto bias = ov::opset8::Constant::create(ov::element::f32, bias_shape, b_data);
        auto add = std::make_shared<ov::opset8::Add>(conv, bias);
        origin_model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{});
    }
    double no_dev, template_dev, cpu_dev;
    // Apply constant folding
    {
        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        no_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_EQ(origin_model->get_ops().size(), cloned_model->get_ops().size());
    }

    // Apply constant folding with extension
    {
        TestExtension<ov::opset8::Convolution, OpModelEvaluate<ov::opset8::Convolution>> test = {};

        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        template_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_NE(origin_model->get_ops().size(), cloned_model->get_ops().size());
        EXPECT_EQ(2, cloned_model->get_ops().size());
    }

    // Apply constant folding with extension on CPU
    {
        get_dev_name() = "CPU";
        TestExtension<ov::opset8::Convolution, OpModelEvaluate<ov::opset8::Convolution>> test = {};

        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        cpu_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_NE(origin_model->get_ops().size(), cloned_model->get_ops().size());
        EXPECT_EQ(2, cloned_model->get_ops().size());
    }

    std::cout << "Duration no_constant_folding: " << no_dev << std::endl;
    std::cout << "Duration template_folding: " << template_dev << std::endl;
    std::cout << "Duration cpu_folding: " << cpu_dev << std::endl;
}

TEST(extension, constant_fold_reshape) {
    const auto& gen_data = [](const ov::Shape& shape) {
        size_t size = ov::shape_size(shape);
        std::vector<float> data(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
        return data;
    };
    std::shared_ptr<ov::Model> origin_model;
    {
        ov::Shape data_shape{1, 7, 32};
        ov::Shape weights_shape{2};
        std::vector<float> in_data = gen_data(data_shape);
        std::vector<int64_t> reshape_data = {-1, 7};
        auto const_data = ov::opset8::Constant::create(ov::element::f32, data_shape, in_data);
        auto shape = ov::opset8::Constant::create(ov::element::i64, weights_shape, reshape_data);
        auto reshape = std::make_shared<ov::opset8::Reshape>(const_data, shape, false);
        origin_model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{});
    }
    double no_dev, template_dev, cpu_dev;
    // Apply constant folding
    {
        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        no_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_NE(origin_model->get_ops().size(), cloned_model->get_ops().size());
        EXPECT_EQ(2, cloned_model->get_ops().size());
    }

    // Apply constant folding with extension
    {
        TestExtension<ov::opset8::Reshape, OpModelEvaluate<ov::opset8::Reshape>> test = {};

        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        template_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_NE(origin_model->get_ops().size(), cloned_model->get_ops().size());
        EXPECT_EQ(2, cloned_model->get_ops().size());
    }

    // Apply constant folding with extension on CPU
    {
        get_dev_name() = "CPU";
        TestExtension<ov::opset8::Reshape, OpModelEvaluate<ov::opset8::Reshape>> test = {};

        auto cloned_model = ov::clone_model(*origin_model);
        ov::pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::ConstantFolding>();
        auto startTime = Time::now();
        pass_manager.run_passes(cloned_model);
        cpu_dev = get_duration_ms_till_now(startTime);

        // Model shouldn't folded
        EXPECT_NE(origin_model->get_ops().size(), cloned_model->get_ops().size());
        EXPECT_EQ(2, cloned_model->get_ops().size());
    }

    std::cout << "Duration no_constant_folding: " << no_dev << std::endl;
    std::cout << "Duration template_folding: " << template_dev << std::endl;
    std::cout << "Duration cpu_folding: " << cpu_dev << std::endl;
}
