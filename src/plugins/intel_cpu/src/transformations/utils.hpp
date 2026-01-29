// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
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
#endif

namespace ov::intel_cpu {

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
template <class T>
bool match_conv_add_mul_fq(const std::shared_ptr<const ov::Node>& node);

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node);

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                          const ov::Strides& strides,
                                          const ov::Shape& kernel_shape,
                                          size_t oc_ic_limit);

bool match_conv_mul_add(const std::shared_ptr<const ov::Node>& node);
#endif

}  // namespace ov::intel_cpu
