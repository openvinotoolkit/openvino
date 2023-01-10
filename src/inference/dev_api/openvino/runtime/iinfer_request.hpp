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

class OPENVINO_RUNTIME_API IInferRequest : public std::enable_shared_from_this<IInferRequest> {
public:
    IInferRequest(const std::shared_ptr<ov::ICompiledModel>& compiled_model);

    virtual void infer() = 0;
    virtual void start_async();

    virtual void wait();
    virtual bool wait_for(const std::chrono::milliseconds& timeout);

    virtual void cancel();

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const;

    virtual ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor);

    virtual std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors);
    virtual void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors);

    virtual std::vector<ov::VariableState> query_state() const;

    virtual void set_callback(std::function<void(std::exception_ptr)> callback);

    virtual void check_tensors() const;

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const;
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const;

    const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const;

protected:
    struct FoundPort {
        size_t idx;
        enum class Type { NotFound = 0, Input, Output } type;

        bool found() {
            return type != Type::NotFound;
        }
        bool is_input() {
            return type == Type::Input;
        }
        bool is_output() {
            return !is_input();
        }
    };

    FoundPort find_port(const ov::Output<const ov::Node>& port) const;
    void convert_batched_tensors();
    void check_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) const;

    std::vector<ov::Tensor> m_input_tensors;
    std::vector<ov::Tensor> m_output_tensors;
    std::unordered_map<size_t, std::vector<ov::Tensor>> m_batched_tensors;

private:
    std::shared_ptr<ov::ICompiledModel> m_compiled_model;
    std::function<void(std::exception_ptr)> m_callback;
    std::shared_ptr<void> m_so;
};

};  // namespace ov
