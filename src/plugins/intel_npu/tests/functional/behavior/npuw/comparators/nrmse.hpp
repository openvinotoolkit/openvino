// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace tests {
namespace metrics {
class NRMSE {
public:
    explicit NRMSE(double threshold);
    bool operator()(const ov::Tensor& actual, const ov::Tensor& reference) const;

private:
    double m_threshold{};
};

}  // namespace metrics
}  // namespace tests 
}  // namespace npuw
}  // namespace ov
