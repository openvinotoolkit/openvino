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
    explicit TransformationsPipeline(const Config &conf) : config(conf) {}
    void apply(std::shared_ptr<ov::Function> func);

private:
    Config config;
};

}  // namespace CLDNNPlugin
