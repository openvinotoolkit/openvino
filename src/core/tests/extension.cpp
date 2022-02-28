// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <gtest/gtest.h>

#include "openvino/core/evaluate_extension.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/opsets/opset8.hpp"
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
    bool has_evaluate(const std::shared_ptr<const ov::Node>& node) const override {
        return node->get_type_info() == ov::op::v0::Relu::get_type_info_static();
    }
    bool evaluate(const std::shared_ptr<const ov::Node>& node,
                  ov::TensorVector& output_values,
                  const ov::TensorVector& input_values) const override {
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

namespace {

class TestReluEvaluate {
public:
    TestReluEvaluate(const ov::element::Type& el_type,
                     const ov::Shape& shape,
                     const std::vector<int64_t>& in_data,
                     const std::vector<int64_t>& out_data,
                     bool add_extension = false) {
        if (add_extension)
            ov::opset8::Relu::add_extension(std::make_shared<Add1Evaluate>());

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

    ~TestReluEvaluate() {
        ov::get_extensions_for_type(ov::opset8::Relu::get_type_info_static()).clear();
    }

private:
    bool m_success = false;
};

}  // namespace

TEST(extension, evaluate_extension) {
    std::vector<int64_t> orig_data = {-2, -1, 1, 2};
    std::vector<int64_t> ref_orig_data = {0, 0, 1, 2};
    std::vector<int64_t> ref_data = {-1, 0, 2, 3};
    ov::element::Type el_type = ov::element::i64;
    ov::Shape shape({1, 1, 2, 2});
    {
        TestReluEvaluate test_eval(el_type, shape, orig_data, ref_data, true);
        EXPECT_TRUE(test_eval.success());
    }

    {
        TestReluEvaluate test_eval(el_type, shape, orig_data, ref_orig_data, false);
        EXPECT_TRUE(test_eval.success());
    }
}
