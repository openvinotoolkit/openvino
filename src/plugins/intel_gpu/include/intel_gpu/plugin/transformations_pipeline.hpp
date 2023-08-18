// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/device.hpp"

namespace ov {
namespace intel_gpu {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const ExecutionConfig &conf, const cldnn::device_info &device_info)
        : config(conf), device_info(device_info) {}
    void apply(std::shared_ptr<ov::Model> func);

private:
    const ExecutionConfig& config;
    cldnn::device_info device_info;
};

}  // namespace intel_gpu
}  // namespace ov
