// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/pass/pass_config.hpp"

namespace ov {
using RTInfoConfig = typename pass::PassConfig;

OPENVINO_API
void copy_runtime_info(const std::shared_ptr<ov::Node>& from,
                       const std::shared_ptr<ov::Node>& to,
                       const std::shared_ptr<ov::RTInfoConfig> rt_config = nullptr);

OPENVINO_API
void copy_runtime_info(const std::shared_ptr<ov::Node>& from,
                       ov::NodeVector to,
                       const std::shared_ptr<ov::RTInfoConfig> rt_config = nullptr);

OPENVINO_API
void copy_runtime_info(const ov::NodeVector& from,
                       const std::shared_ptr<ov::Node>& to,
                       const std::shared_ptr<ov::RTInfoConfig> rt_config = nullptr);

OPENVINO_API
void copy_runtime_info(const ov::NodeVector& from,
                       ov::NodeVector to,
                       const std::shared_ptr<ov::RTInfoConfig> rt_config = nullptr);

OPENVINO_API
void copy_output_runtime_info(const ov::OutputVector& from,
                              ov::OutputVector to,
                              const std::shared_ptr<ov::RTInfoConfig> rt_config = nullptr);
}  // namespace ov
