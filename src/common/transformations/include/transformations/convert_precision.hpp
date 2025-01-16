// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertPrecision;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertPrecision transformation convert precision for entire ov::Model
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
 * we have created special fuse_type_into_<type> function (can be found in cpp file) that performs type fusion
 * into operation. m_additional_type_to_fuse_map allows to rewrite existing type convertors.
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

struct EnumClassHash {
    template <class T>
    std::size_t operator()(T t) const {
        return static_cast<size_t>(t);
    }
};

using precisions_map = std::unordered_map<ov::element::Type_t, ov::element::Type, EnumClassHash>;
using type_to_fuse_map =
    std::unordered_map<ov::NodeTypeInfo, std::function<bool(const std::shared_ptr<ov::Node>&, const precisions_map&)>>;

class ov::pass::ConvertPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertPrecision", "0");
    ConvertPrecision(ov::element::Type_t from,
                     ov::element::Type_t to,
                     type_to_fuse_map additional_type_to_fuse_map = {},
                     bool keep_precision_sensitive_in_fp32 = false,
                     bool convert_input_output_precision = true,
                     bool store_original_precision_as_rt_attribute = false)
        : m_precisions(precisions_map{{from, to}}),
          m_additional_type_to_fuse_map(std::move(additional_type_to_fuse_map)),
          m_keep_precision_sensitive_in_fp32(keep_precision_sensitive_in_fp32),
          m_convert_input_output_precision(convert_input_output_precision),
          m_store_original_precision_as_rt_attribute(store_original_precision_as_rt_attribute) {}

    ConvertPrecision(const precisions_map& precisions,
                     const type_to_fuse_map& additional_type_to_fuse_map = {},
                     bool keep_precision_sensitive_in_fp32 = false,
                     bool convert_input_output_precision = true,
                     bool store_original_precision_as_rt_attribute = false)
        : m_precisions(precisions),
          m_additional_type_to_fuse_map(additional_type_to_fuse_map),
          m_keep_precision_sensitive_in_fp32(keep_precision_sensitive_in_fp32),
          m_convert_input_output_precision(convert_input_output_precision),
          m_store_original_precision_as_rt_attribute(store_original_precision_as_rt_attribute) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    precisions_map m_precisions;
    type_to_fuse_map m_additional_type_to_fuse_map;
    bool m_keep_precision_sensitive_in_fp32;
    bool m_convert_input_output_precision;
    bool m_store_original_precision_as_rt_attribute;
};
