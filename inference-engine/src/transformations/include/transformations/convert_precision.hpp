// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include <transformations_visibility.hpp>

#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class NGRAPH_API ConvertPrecision;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertPrecision transformation convert precision for entire ngraph::Function
 * List of supported precision conversion:
 *    FROM -> TO
 *      u8 -> i32
 *     u16 -> i32
 *     u32 -> i32
 *     u64 -> i32
 *     i64 -> i32
 *     f16 -> f32
 *    bool -> u8
 *    bool -> i32
 *
 * For all operations from opset1-opset4 this conversions can be applied without adding Conversion operations.
 * That is possible because all operations that produces "FROM" type can produce "TO" type. And for this operations
 * we have created special fuse_type_into_<type> functoin (can be found in cpp file) that performs type fusion
 * into operation.
 *
 * List of operations that are supported by this transformations for i64 -> i32 conversion:
 *     opset4::Parameter
 *     opset4::Convert
 *     opset4::ShapeOf
 *     opset4::Range
 *     opset3::NonMaxSuppression
 *     opset4::NonMaxSuppression
 *     opset4::TopK
 *     opset4::NonZero
 *     opset4::Bucketize
 *
 * List of operations that are supported by this transformations for bool -> u8 conversion:
 *     LogicalAnd
 *     LogicalNot
 *     LogicalOr
 *     LogicalXor
 *     ReduceLogicalAnd
 *     ReduceLogicalOr
 *     Equal
 *     NotEqual
 *     Greater
 *     GreaterEqual
 *     Less
 *     LessEqual
 */

using type_to_fuse_map = std::unordered_map<ngraph::NodeTypeInfo, std::function<bool(const std::shared_ptr<ngraph::Node>&, ngraph::element::Type, size_t idx)>>;
using precisions_array = std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>>;

class ngraph::pass::ConvertPrecision : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertPrecision(ngraph::element::Type_t from, ngraph::element::Type_t to, type_to_fuse_map additional_type_to_fuse_map = {})
        : FunctionPass(),
        m_precisions(precisions_array {{ from, to }}),
        m_additional_type_to_fuse_map(additional_type_to_fuse_map) {}

    ConvertPrecision(const precisions_array& precisions, const type_to_fuse_map & additional_type_to_fuse_map = {})
        : FunctionPass(),
        m_precisions(precisions),
        m_additional_type_to_fuse_map(additional_type_to_fuse_map) {}

    bool run_on_function(std::shared_ptr<Function> f) override;
private:
    precisions_array m_precisions;
    type_to_fuse_map m_additional_type_to_fuse_map;
};
