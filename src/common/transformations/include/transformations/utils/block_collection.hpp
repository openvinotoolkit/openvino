// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "openvino/pass/pattern/op/block.hpp"

namespace ov::pass::pattern::blocks {

std::shared_ptr<ov::Node> l2_norm_block(const ov::Output<ov::Node>& input);
std::shared_ptr<ov::Node> dq_constant_block();
std::shared_ptr<ov::Node> attention_mask();
std::shared_ptr<ov::Node> qkv_projection_block(const ov::Output<ov::Node>& input);
std::shared_ptr<ov::Node> sdpa_preprocessing_block(const ov::Output<ov::Node>& input);
std::shared_ptr<ov::Node> sdpa_block(const ov::Output<ov::Node>& q,
                                     const ov::Output<ov::Node>& k,
                                     const ov::Output<ov::Node>& v);
std::shared_ptr<ov::Node> post_sdpa_projection_block(const ov::Output<ov::Node>& qkv);

}  // namespace ov::pass::pattern::blocks