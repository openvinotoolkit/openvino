// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/function.hpp>

#include "intel_gpu/plugin/device_config.hpp"

namespace ov {
namespace intel_gpu {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const Config &conf, const cldnn::device_info &device_info)
        : config(conf), device_info(device_info) {}
    void apply(std::shared_ptr<ov::Model> func);

private:
    Config config;
    cldnn::device_info device_info;
};

}  // namespace intel_gpu
}  // namespace ov
