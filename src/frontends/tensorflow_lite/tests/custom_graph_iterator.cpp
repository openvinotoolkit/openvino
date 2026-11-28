// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "tf_utils.hpp"

using namespace ov::frontend::tensorflow_lite;

namespace {

// A minimal custom GraphIterator for the model ADD(input, weight) -> output. The weight constant is
// supplied through TensorMetaInfo::m_tensor_data with m_tensor_data_size left at its default of 0,
// which is how a GraphIterator implementation that predates the size field reports a constant.
// load_model() must keep accepting such a constant (its buffer size check treats 0 as "unknown").

TensorMetaInfo make_tensor_info(const std::string& name,
                                const ov::PartialShape& shape,
                                const ov::element::Type& type,
                                const uint8_t* data = nullptr) {
    TensorMetaInfo info;
    info.m_partial_shape = shape;
    info.m_element_type = type;
    info.m_tensor_name = name;
    info.m_tensor_data = data;
    return info;
}

class TensorDecoder : public DecoderBaseTensor {
public:
    TensorDecoder(TensorMetaInfo info, int64_t input_idx, int64_t output_idx)
        : m_info(std::move(info)),
          m_input_idx(input_idx),
          m_output_idx(output_idx) {}

    TensorMetaInfo get_tensor_info() const override {
        return m_info;
    }
    int64_t get_input_idx() const override {
        return m_input_idx;
    }
    int64_t get_output_idx() const override {
        return m_output_idx;
    }
    ov::Any get_attribute(const std::string&) const override {
        return {};
    }
    size_t get_input_size() const override {
        return 0;
    }
    void get_input_node(size_t, std::string&, std::string&, size_t&) const override {}
    const std::string& get_op_type() const override {
        return m_type;
    }
    const std::string& get_op_name() const override {
        return m_info.m_tensor_name;
    }

private:
    TensorMetaInfo m_info;
    int64_t m_input_idx;
    int64_t m_output_idx;
    std::string m_type = "Tensor";
};

class AddDecoder : public DecoderBaseOperation {
public:
    AddDecoder(std::vector<TensorMetaInfo> inputs, TensorMetaInfo output)
        : m_inputs(std::move(inputs)),
          m_output(std::move(output)) {}

    std::string get_input_tensor_name(size_t idx) const override {
        return m_inputs.at(idx).m_tensor_name;
    }
    ov::element::Type get_input_tensor_type(size_t idx) const override {
        return m_inputs.at(idx).m_element_type;
    }
    std::string get_output_tensor_name(size_t idx) const override {
        return get_output_tensor_info(idx).m_tensor_name;
    }
    ov::element::Type get_output_tensor_type(size_t idx) const override {
        return get_output_tensor_info(idx).m_element_type;
    }
    TensorMetaInfo get_input_tensor_info(size_t idx) const override {
        return m_inputs.at(idx);
    }
    TensorMetaInfo get_output_tensor_info(size_t idx) const override {
        FRONT_END_GENERAL_CHECK(idx == 0, "Requested output is out-of-range");
        return m_output;
    }
    size_t get_output_size() const override {
        return 1;
    }
    ov::Any get_attribute(const std::string& name) const override {
        if (name == "fused_activation_function")
            return std::string("NONE");
        return {};
    }
    size_t get_input_size() const override {
        return m_inputs.size();
    }
    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string&,
                        size_t& producer_output_port_index) const override {
        producer_name = m_inputs.at(input_port_idx).m_tensor_name;
        producer_output_port_index = 0;
    }
    const std::string& get_op_type() const override {
        return m_type;
    }
    const std::string& get_op_name() const override {
        return m_name;
    }

private:
    std::vector<TensorMetaInfo> m_inputs;
    TensorMetaInfo m_output;
    std::string m_type = "ADD";
    std::string m_name = "add";
};

class SingleOpGraphIterator : public GraphIterator {
public:
    explicit SingleOpGraphIterator(std::vector<std::shared_ptr<DecoderBase>> nodes) : m_nodes(std::move(nodes)) {}

    size_t size() const override {
        return m_nodes.size();
    }
    void reset() override {
        m_index = 0;
    }
    void next() override {
        ++m_index;
    }
    bool is_end() const override {
        return m_index >= m_nodes.size();
    }
    std::shared_ptr<DecoderBase> get_decoder() const override {
        return m_nodes.at(m_index);
    }
    size_t get_subgraph_size() const override {
        return 0;
    }
    std::shared_ptr<GraphIterator> get_subgraph(size_t) const override {
        return nullptr;
    }

private:
    std::vector<std::shared_ptr<DecoderBase>> m_nodes;
    size_t m_index = 0;
};

}  // namespace

TEST(TensorFlowLiteCustomGraphIterator, constant_with_default_data_size_loads) {
    const std::vector<int32_t> weight_values{1, 2, 3, 4};
    const ov::PartialShape shape{1, 4};
    const auto input = make_tensor_info("input", shape, ov::element::i32);
    const auto weight =
        make_tensor_info("weight", shape, ov::element::i32, reinterpret_cast<const uint8_t*>(weight_values.data()));
    const auto output = make_tensor_info("output", shape, ov::element::i32);
    ASSERT_EQ(weight.m_tensor_data_size, 0u);  // the public default, deliberately left untouched

    std::vector<std::shared_ptr<DecoderBase>> nodes{
        std::make_shared<TensorDecoder>(input, 0, -1),
        std::make_shared<TensorDecoder>(output, -1, 0),
        std::make_shared<AddDecoder>(std::vector<TensorMetaInfo>{input, weight}, output)};
    GraphIterator::Ptr iterator = std::make_shared<SingleOpGraphIterator>(nodes);

    ov::frontend::FrontEndManager fem;
    auto fe = fem.load_by_framework(TF_LITE_FE);
    ASSERT_NE(fe, nullptr);
    ov::frontend::InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = fe->load(iterator));
    ASSERT_NE(input_model, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = fe->convert(input_model));
    ASSERT_NE(model, nullptr);

    std::shared_ptr<ov::op::v0::Constant> constant;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto candidate = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
            if (candidate->get_friendly_name() == "weight")
                constant = candidate;
        }
    }
    ASSERT_NE(constant, nullptr) << "weight constant not found in the converted model";
    EXPECT_EQ(constant->get_shape(), ov::Shape({1, 4}));
    EXPECT_EQ(constant->get_vector<int32_t>(), weight_values);
}
