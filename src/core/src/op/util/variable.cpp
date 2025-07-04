// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/variable.hpp"

namespace ov {
AttributeAdapter<std::shared_ptr<op::util::Variable>>::~AttributeAdapter() = default;
}  // namespace ov
