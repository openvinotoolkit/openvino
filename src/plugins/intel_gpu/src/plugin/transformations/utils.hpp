// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/type_relaxed.hpp"

namespace ov {
namespace intel_gpu {

template<typename T, typename... Args>
std::shared_ptr<T> make_type_relaxed(const element::TypeVector& input_data_types,
                                     const element::TypeVector& output_data_types,
                                     Args&&... args) {
    return std::make_shared<ov::op::TypeRelaxed<T>>(std::forward<Args>(args)...);
}

}  // namespace intel_gpu
}  // namespace ov
