// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime InferRequest interface
 * @file iinfer_request.hpp
 */

#pragma once

#include <exception>
#include <memory>
#include <openvino/runtime/tensor.hpp>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {

class OPENVINO_RUNTIME_API ISyncInferRequest : public IInferRequest {
public:
    ISyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model);

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;

    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const override;
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) override;
    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors);

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override;

    const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const override;

protected:
    struct FoundPort {
        size_t idx;
        enum class Type { NOT_FOUND = 0, INPUT, OUTPUT } type;

        bool found() {
            return type != Type::NOT_FOUND;
        }
        bool is_input() {
            return type == Type::INPUT;
        }
        bool is_output() {
            return !is_input();
        }
    };

    FoundPort find_port(const ov::Output<const ov::Node>& port) const;
    void convert_batched_tensors();
    void check_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) const;

    void check_tensors() const override;

    std::vector<ov::Tensor> m_input_tensors;
    std::vector<ov::Tensor> m_output_tensors;
    std::unordered_map<size_t, std::vector<ov::Tensor>> m_batched_tensors;

private:
    std::shared_ptr<ov::ICompiledModel> m_compiled_model;
};

};  // namespace ov
