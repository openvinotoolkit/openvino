// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "kernel_selector_common.h"
#include "tensor_type.h"

namespace cldnn {

kernel_selector::Datatype to_data_type(data_types data_type);
kernel_selector::DataLayout to_data_layout(format data_format);
kernel_selector::Tensor::NDims compute_tensor_dimensions(const layout& tensor_layout,
                                                         size_t channel_count,
                                                         tensor view_offset = tensor{});
kernel_selector::DataTensor convert_data_tensor(const layout& tensor_layout, tensor view_offset = tensor{});

}  // namespace cldnn
