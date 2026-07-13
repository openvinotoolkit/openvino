// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/op/constant.hpp"

using namespace ov::frontend::onnx;

namespace {

// Minimal DecoderBaseTensor exposing a single graph output tensor whose data lives entirely in
// TensorMetaInfo::m_tensor_data_any (as either const std::string* or std::vector<std::string>),
// with no raw m_tensor_data pointer set. Used to drive that data through the full ONNX FE
// pipeline without needing an ONNX protobuf model.
class SingleTensorDecoder : public DecoderBaseTensor {
public:
    explicit SingleTensorDecoder(TensorMetaInfo info) : m_info(std::move(info)) {}

    const TensorMetaInfo& get_tensor_info() const override {
        return m_info;
    }
    int64_t get_input_idx() const override {
        return -1;
    }
    int64_t get_output_idx() const override {
        return 0;
    }

    ov::Any get_attribute(const std::string&) const override {
        return {};
    }
    size_t get_input_size() const override {
        return 0;
    }
    void get_input_node(size_t, std::string&, std::string&, size_t&) const override {}
    const std::string& get_op_type() const override {
        static const std::string type = "Tensor";
        return type;
    }
    const std::string& get_op_name() const override {
        static const std::string name;
        return name;
    }

private:
    TensorMetaInfo m_info;
};

// GraphIterator yielding exactly one DecoderBaseTensor (a graph output with inline data) and no
// operation nodes.
class SingleTensorGraphIterator : public GraphIterator {
public:
    explicit SingleTensorGraphIterator(std::shared_ptr<DecoderBase> decoder) : m_decoder(std::move(decoder)) {}

    size_t size() const override {
        return 1;
    }
    void reset() override {
        m_done = false;
    }
    void next() override {
        m_done = true;
    }
    bool is_end() const override {
        return m_done;
    }
    std::shared_ptr<DecoderBase> get_decoder() const override {
        return m_done ? nullptr : m_decoder;
    }
    int64_t get_opset_version(const std::string&) const override {
        return 1;
    }
    std::map<std::string, std::string> get_metadata() const override {
        return {};
    }
    std::filesystem::path get_model_dir() const override {
        return {};
    }

private:
    std::shared_ptr<DecoderBase> m_decoder;
    bool m_done = false;
};

std::shared_ptr<ov::Model> convert_single_string_tensor(TensorMetaInfo info) {
    auto decoder = std::make_shared<SingleTensorDecoder>(std::move(info));
    auto iterator = std::make_shared<SingleTensorGraphIterator>(decoder);
    auto graph_iter = std::dynamic_pointer_cast<GraphIterator>(iterator);

    auto frontend = ov::frontend::FrontEndManager().load_by_framework("onnx");
    EXPECT_NE(frontend, nullptr);
    EXPECT_TRUE(frontend->supported(graph_iter));

    auto input_model = frontend->load(graph_iter);
    EXPECT_NE(input_model, nullptr);
    return frontend->convert(input_model);
}

}  // namespace

// Drives a STRING output tensor backed by `const std::string*` (the new zero-copy path) through
// the full ONNX FE pipeline via a custom GraphIterator/DecoderBaseTensor, and verifies that
// Tensor::get_data<std::string>() correctly materializes it into an ov::op::v0::Constant. This
// also exercises the create_const_or_param() get_data_any() dispatch fix in translate_session.cpp:
// without it, this tensor would be misclassified as a Parameter instead of a Constant.
TEST(StringTensorExternalData, ConstStringPointerPath_ThroughConvert) {
    static const std::string tensor_name = "str_out";
    std::string backing_strings[3] = {"hello", "world", "test"};

    TensorMetaInfo info;
    info.m_partial_shape = ov::PartialShape{3};
    info.m_element_type = ov::element::string;
    info.m_tensor_data = nullptr;
    info.m_tensor_data_any = static_cast<const std::string*>(backing_strings);
    info.m_tensor_data_size = 3;
    info.m_tensor_name = &tensor_name;
    info.m_is_raw = false;

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = convert_single_string_tensor(info));
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(model->get_results().size(), 1u);
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::string);
    EXPECT_EQ(constant->get_value_strings(), (std::vector<std::string>{"hello", "world", "test"}));
}

// Same as above but exercises the pre-existing std::vector<std::string> storage path through the
// same convert() pipeline, as a backward-compatibility counterpart to the test above.
TEST(StringTensorExternalData, VectorStringPath_ThroughConvert) {
    static const std::string tensor_name = "str_out";
    std::vector<std::string> backing_strings = {"foo", "bar", "baz"};

    TensorMetaInfo info;
    info.m_partial_shape = ov::PartialShape{3};
    info.m_element_type = ov::element::string;
    info.m_tensor_data = nullptr;
    info.m_tensor_data_any = backing_strings;
    info.m_tensor_data_size = 3;
    info.m_tensor_name = &tensor_name;
    info.m_is_raw = false;

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = convert_single_string_tensor(info));
    ASSERT_NE(model, nullptr);

    ASSERT_EQ(model->get_results().size(), 1u);
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::string);
    EXPECT_EQ(constant->get_value_strings(), (std::vector<std::string>{"foo", "bar", "baz"}));
}
