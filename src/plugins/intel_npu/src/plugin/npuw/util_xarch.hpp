// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/op/constant.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {
namespace XARCH {

void unpack(const ov::SoPtr<ov::ITensor>& from, const ov::SoPtr<ov::ITensor>& to);

void unpack_scale(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& scale,
                  const ov::SoPtr<ov::ITensor>& to);

void unpack_scale_zp(const ov::SoPtr<ov::ITensor>& from,
                     const ov::SoPtr<ov::ITensor>& zerop,
                     const ov::SoPtr<ov::ITensor>& scale,
                     const ov::SoPtr<ov::ITensor>& to);

ov::Tensor to_f16(const ov::Tensor& t);

}  // namespace XARCH
}  // namespace util
}  // namespace npuw
}  // namespace ov
