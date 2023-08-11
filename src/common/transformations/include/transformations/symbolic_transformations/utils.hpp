// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

bool get_labels(const ov::PartialShape& shape, ov::TensorLabel& labels);
bool get_labels(const ov::Output<ov::Node>& output, ov::TensorLabel& labels);

bool are_unique_and_equal_labels(const ov::TensorLabel& lhs, const ov::TensorLabel& rhs);

bool labels_eq_or_eq_static_dims(const ov::Dimension& lhs, const ov::Dimension& rhs);

bool last_two_dims_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs);

bool equalize_two_last_dims(const ov::PartialShape& from, ov::PartialShape& to);

bool reshape_keeps_last_two_dims(const std::shared_ptr<ov::Node>& op);

bool batches_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs, bool one_dim_can_differ = false);

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1);

ov::Output<ov::Node> get_shape_from_sources(const ov::Output<ov::Node>& batch_dims_source,
                                            const ov::Output<ov::Node>& non_batch_dims_source);

int64_t get_idx_of_label_in_source(const ov::Output<ov::Node>& source, const ov::label_t& label);

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_idx(const ov::Output<ov::Node>& source,
                                                                         const ov::element::Type& et,
                                                                         const ov::Shape& shape,
                                                                         const int64_t& idx);

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_label(const ov::Output<ov::Node>& source,
                                                                           const ov::element::Type& et,
                                                                           const ov::Shape& shape,
                                                                           const ov::label_t& label);

void optimize_value_usage(ov::Output<ov::Node>& output,
                          std::unordered_map<ov::label_t, ov::Output<ov::Node>>& label_shape_source,
                          std::unordered_map<ov::label_t, ov::Output<ov::Node>>& label_value_source);

void save_shape_sources(const ov::Output<ov::Node>& output,
                        std::unordered_map<ov::label_t, ov::Output<ov::Node>>& label_shape_source);
