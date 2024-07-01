// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/core/model.hpp"

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device.hpp"

namespace ov {
namespace intel_gpu {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const ExecutionConfig &conf, const std::shared_ptr<RemoteContextImpl>& context)
        : config(conf), m_context(context), device_info(context->get_engine().get_device_info()) {}
    void apply(std::shared_ptr<ov::Model> func);

private:
    const ExecutionConfig& config;
    std::shared_ptr<RemoteContextImpl> m_context;
    cldnn::device_info device_info;
};

}  // namespace intel_gpu
}  // namespace ov
