// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/smart_reshape/set_batch_size.hpp"
#include "transformations/smart_reshape/smart_reshape.hpp"
#include "transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp"
#include "transformations/fix_rt_info.hpp"

std::atomic<size_t> ov::Node::m_next_instance_id(0);
BWDCMP_RTTI_DEFINITION(ov::op::v0::Parameter);
BWDCMP_RTTI_DEFINITION(ov::op::v0::Constant);
BWDCMP_RTTI_DEFINITION(ov::op::v5::NonMaxSuppression);
BWDCMP_RTTI_DEFINITION(ngraph::pass::ConvertMatrixNmsToMatrixNmsIE);
BWDCMP_RTTI_DEFINITION(ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE);
BWDCMP_RTTI_DEFINITION(ngraph::pass::FixRtInfo);
BWDCMP_RTTI_DEFINITION(ngraph::pass::SetBatchSize);
BWDCMP_RTTI_DEFINITION(ngraph::pass::SmartReshape);
BWDCMP_RTTI_DEFINITION(ov::pass::RemoveConcatZeroDimInput);
BWDCMP_RTTI_DEFINITION(ov::pass::RemoveMultiSubGraphOpDanglingParams);
BWDCMP_RTTI_DEFINITION(ngraph::pass::DisableConvertConstantFoldingOnConstPath);