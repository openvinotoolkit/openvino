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
#include "openvino/runtime/profiling_info.hpp"

namespace ov {

class IAsyncInferRequest;
class ICompiledModel;

class OPENVINO_RUNTIME_API IInferRequest {
public:
    virtual ~IInferRequest();
    virtual void infer() = 0;

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const = 0;

    virtual ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const = 0;
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) = 0;

    virtual std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const = 0;
    virtual void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) = 0;

    virtual std::vector<ov::VariableState> query_state() const = 0;

    virtual const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const = 0;

    virtual const std::vector<ov::Output<const ov::Node>>& get_inputs() const = 0;
    virtual const std::vector<ov::Output<const ov::Node>>& get_outputs() const = 0;

protected:
    virtual void check_tensors() const = 0;
    friend IAsyncInferRequest;
};

};  // namespace ov

