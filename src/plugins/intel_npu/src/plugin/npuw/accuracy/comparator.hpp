// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../common.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace metrics {
class NRMSE {
public:
    explicit NRMSE(double threshold);
    bool operator()(const ov::SoPtr<ov::ITensor>& backup_tensor, const ov::SoPtr<ov::ITensor>& original_tensor) const;

private:
    double m_threshold{};
};

}  // namespace metrics
}  // namespace npuw
}  // namespace ov
