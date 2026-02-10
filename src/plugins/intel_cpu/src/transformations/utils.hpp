// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

template <class T>
bool match_conv_mul_add_fq(const std::shared_ptr<const ov::Node>& node);

enum class FQMulAddPattern : std::uint8_t { ConvMulAdd, ConvAddMul };

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node, FQMulAddPattern pattern);

bool match_conv_fq_same_types(const std::shared_ptr<const ov::Node>& node);

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<int64_t>& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit);

bool match_conv_mul_add(const std::shared_ptr<const ov::Node>& node);

}  // namespace ov::intel_cpu
