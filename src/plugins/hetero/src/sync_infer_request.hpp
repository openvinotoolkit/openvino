// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/itt.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace hetero {

class CompiledModel;
class AsyncInferRequest;

class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model);

    ~InferRequest();

    void infer() override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;

    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void check_tensors() const override;

private:
    friend class AsyncInferRequest;

    ov::SoPtr<ov::IAsyncInferRequest> get_request(const ov::Output<const ov::Node>& port) const;

    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> m_subrequests;
    std::map<ov::Output<const ov::Node>, size_t> m_port_to_subrequest_idx;
};

}  // namespace hetero
}  // namespace ov
