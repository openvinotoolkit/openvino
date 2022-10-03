// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"

std::atomic<size_t> ov::Node::m_next_instance_id(0);
BWDCMP_RTTI_DEFINITION(ov::op::v0::Parameter);
BWDCMP_RTTI_DEFINITION(ov::op::v0::Constant);
BWDCMP_RTTI_DEFINITION(ov::op::v5::NonMaxSuppression);
BWDCMP_RTTI_DEFINITION(ngraph::pass::ConvertMatrixNmsToMatrixNmsIE);
BWDCMP_RTTI_DEFINITION(ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE);