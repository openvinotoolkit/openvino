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
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {

class IAsyncInferRequest;

class OPENVINO_RUNTIME_API ISyncInferRequest : public std::enable_shared_from_this<ISyncInferRequest> {
public:
    ISyncInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model);

    virtual void infer() = 0;

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const = 0;

    virtual ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor);

    virtual std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors);
    virtual void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors);

    virtual std::vector<ov::VariableState> query_state() const = 0;

    virtual void set_callback(std::function<void(std::exception_ptr)> callback) = 0;

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const;
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const;

    const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const;

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

    virtual void check_tensors() const;

    std::vector<ov::Tensor> m_input_tensors;
    std::vector<ov::Tensor> m_output_tensors;
    std::unordered_map<size_t, std::vector<ov::Tensor>> m_batched_tensors;

private:
    std::shared_ptr<ov::ICompiledModel> m_compiled_model;
    std::function<void(std::exception_ptr)> m_callback;
    friend IAsyncInferRequest;
};

};  // namespace ov
