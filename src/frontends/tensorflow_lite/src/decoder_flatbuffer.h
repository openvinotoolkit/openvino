// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "graph_iterator_flatbuffer.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"
#include "tensor_lite_place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TensorLitePlace;
struct TensorInfo;

class DecoderFlatBuffer : public ov::frontend::tensorflow_lite::DecoderBaseOperation {
public:
    explicit DecoderFlatBuffer(const tflite::Operator* node_def,
                               const std::string& type,
                               const std::string& name,
                               std::map<size_t, ov::frontend::tensorflow_lite::TensorInfo> input_info,
                               std::map<size_t, ov::frontend::tensorflow_lite::TensorInfo> output_info)
        : m_node_def(node_def),
          m_type(type),
          m_name(name),
          m_input_info(input_info),
          m_output_info(output_info) {}

    template <class Ret, class Class>
    Ret get_attribute(Ret (Class::*member)() const) const {
        const auto opts = m_node_def->builtin_options_as<Class>();
        FRONT_END_GENERAL_CHECK(opts != nullptr, "Chosen Builtin Option is not accessible for this node");
        return (opts->*member)();
    }

    template <class Ret, class Class>
    bool has_attribute(Ret (Class::*member)() const) const {
        const auto opts = m_node_def->builtin_options_as<Class>();
        if (opts == nullptr)
            return false;
        return (opts->*member)();
    }

    ov::Any get_attribute(const std::string& name) const override;

    size_t get_input_size() const override;
    size_t get_output_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;

    std::string get_output_tensor_name(size_t idx) const override;
    element::Type get_output_tensor_type(size_t idx) const override;
    std::string get_input_tensor_name(size_t idx) const override;
    ov::element::Type get_input_tensor_type(size_t idx) const override;

    TensorMetaInfo get_input_tensor_info(size_t idx) const override;
    TensorMetaInfo get_output_tensor_info(size_t idx) const override;

    const std::string& get_op_type() const override;
    const std::string& get_op_name() const override;

    std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_input_tensor(
        size_t idx,
        const ov::frontend::InputModel& model) const;

    std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_output_tensor(
        size_t idx,
        const ov::frontend::InputModel& model) const;

    virtual ~DecoderFlatBuffer() = default;

protected:
    std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> decode_tensor(
        const ov::frontend::tensorflow_lite::TensorInfo& tensor_info,
        const ov::frontend::InputModel& model) const;

    const tflite::Operator* m_node_def;
    std::string m_type, m_name;
    std::map<size_t, ov::frontend::tensorflow_lite::TensorInfo> m_input_info, m_output_info;
};

class DecoderFlatBufferTensors : public DecoderBaseTensor {
public:
    DecoderFlatBufferTensors(const TensorInfo& tensor_info, int64_t input_idx, int64_t output_idx);

    TensorMetaInfo get_tensor_info() const override {
        return m_tensor_meta_info;
    }

    int64_t get_input_idx() const override {
        return m_input_idx;
    }

    int64_t get_output_idx() const override {
        return m_output_idx;
    }

    ov::Any get_attribute(const std::string& name) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_attribute");
    }

    size_t get_input_size() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_size");
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_input_node");
    }

    const std::string& get_op_type() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_type");
    }

    const std::string& get_op_name() const override {
        FRONT_END_NOT_IMPLEMENTED("DecoderFlatBufferTensors::get_op_name");
    }

private:
    TensorMetaInfo m_tensor_meta_info;
    int64_t m_input_idx, m_output_idx;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
