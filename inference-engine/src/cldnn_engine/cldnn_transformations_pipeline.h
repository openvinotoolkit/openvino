// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/function.hpp>

#include "cldnn_config.h"

namespace CLDNNPlugin {

class TransformationsPipeline {
public:
    explicit TransformationsPipeline(const Config &conf, const cldnn::device_info &device_info)
        : config(conf), device_info(device_info) {}
    void apply(std::shared_ptr<ov::Function> func);

private:
    Config config;
    cldnn::device_info device_info;
};

}  // namespace CLDNNPlugin
