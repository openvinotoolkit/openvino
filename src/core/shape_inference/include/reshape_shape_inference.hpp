// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace reshape {

// helper to hold dimension products.
template <class TDim>
struct Product {
    TDim not_labeled;
    TDim labeled;

    TDim total() const {
        return not_labeled * labeled;
    }
};

/**
 * @brief Check if pattern shape has `special_zero` at specified dimension.
 *
 * @tparam TShape Type of shape.
 *
 * @param pattern  Shape pattern to check.
 * @param idx      Dimension index in the shape pattern.
 * @return true    True if `special_zero` found otherwise false.
 */
template <class TShape>
bool has_pattern_special_zero_at(const TShape& pattern, size_t idx) {
    return (idx < pattern.size() && pattern[idx] == 0);
}

// Resolve input products for PartialShape
template <class TShape,
          class TDim = typename TShape::value_type,
          typename std::enable_if<std::is_same<TShape, PartialShape>::value>::type* = nullptr>
Product<TDim> resolve_input_product(const TShape& input_shape, const TShape& pattern_shape, bool special_zero) {
    auto in_products = Product<TDim>{1, 1};
    bool label_found = false;

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (!special_zero || !reshape::has_pattern_special_zero_at(pattern_shape, i)) {
            if (DimensionTracker::get_label(input_shape[i]) == no_label) {
                in_products.not_labeled *= input_shape[i];
            } else if (label_found) {
                in_products.labeled *= input_shape[i];
            } else {
                in_products.labeled = input_shape[i];
                label_found = true;
            }
        }
    }

    return in_products;
}

// Resolve input products for StaticShape
template <class TShape,
          class TDim = typename TShape::value_type,
          typename std::enable_if<!std::is_same<TShape, PartialShape>::value>::type* = nullptr>
Product<TDim> resolve_input_product(const TShape& input_shape,
                                    const result_shape_t<TShape>& pattern_shape,
                                    bool special_zero) {
    auto in_products = Product<TDim>{1, 1};

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (!special_zero || !reshape::has_pattern_special_zero_at(pattern_shape, i)) {
            in_products.not_labeled *= input_shape[i];
        }
    }

    return in_products;
}

// resolve minus one dimension for ov::Dimension
template <class TDim,
          typename std::enable_if<std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
TDim resolve_minus_one_dim(const Product<TDim>& product_in, const TDim& product_out) {
    TDim out = product_in.labeled;

    if (product_out.is_static() && product_in.not_labeled.is_static()) {
        if (product_in.not_labeled != product_out) {
            out *= product_in.not_labeled;
            out /= product_out.get_length();
        }
    } else {
        using namespace ov::util;
        out *= product_in.not_labeled;
        auto& out_interval = out.get_interval();
        if (product_out.get_min_length() != 0 && out_interval != Interval{} && product_out != TDim{}) {
            out_interval.set_max_val(out_interval.get_max_val() / product_out.get_min_length());
        } else {
            out_interval.set_max_val(Interval::s_max);
        }
        if (product_out.get_max_length() != 0) {
            out_interval.set_min_val(ceil_div(out_interval.get_min_val(), product_out.get_interval().get_max_val()));
        }
    }
    return out;
}

// resolve minus one dimension for static dimension
template <class TDim,
          typename std::enable_if<!std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
TDim resolve_minus_one_dim(const Product<TDim>& product_in, const TDim& product_out) {
    return product_in.total() / product_out.get_length();
}

/**
 * @brief Get the pattern and minus one idx from input bounds.
 *
 * @param op      Pointer to reshape node.
 * @param bounds  Vector of reshape pattern bounds.
 *
 * @return Pair which got bounds converted to shape and `minus_one` index in pattern (-1 if not found).
 */
template <class TShape>
std::pair<TShape, int64_t> get_pattern_and_minus_one_idx(const Node* const op,
                                                         const std::vector<std::pair<int64_t, int64_t>>& bounds) {
    using namespace ov::util;
    const auto minus_one_bound = std::make_pair(dim::inf_bound, dim::inf_bound);

    auto result = std::make_pair(TShape{}, dim::inf_bound);
    auto& shape = std::get<0>(result);
    shape.reserve(bounds.size());

    auto& minus_one_idx = std::get<1>(result);
    auto bounds_iter = bounds.begin();

    for (size_t i = 0; i < bounds.size(); ++i, ++bounds_iter) {
        if (*bounds_iter == minus_one_bound) {
            NODE_VALIDATION_CHECK(op, minus_one_idx == dim::inf_bound, "More than one dimension has size of -1");
            minus_one_idx = static_cast<int64_t>(i);
        }
        NODE_VALIDATION_CHECK(op, *bounds_iter >= minus_one_bound, "Dim size cannot be less than -1");
        shape.emplace_back(bounds_iter->first, bounds_iter->second);
    }

    return result;
}

/**
 * @brief Set the pattern labels on pattern shape if this input is labeled.
 *
 * @param op     Pointer to reshape node.
 * @param shape  Pointer to shape for labels set.
 */
template <class TShape, typename std::enable_if<std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_labels(const Node* const op, TShape& shape) {
    if (op->get_input_size() > 0) {
        auto labels = op->get_input_source_output(1).get_tensor().get_value_label();

        if (!labels.empty()) {
            auto label_iter = labels.begin();
            for (auto& d : shape) {
                if (*label_iter != no_label) {
                    DimensionTracker::set_label(d, *label_iter);
                }
                ++label_iter;
            }
        }
    }
}

/** @brief Shapes other than PartialShape have no labels. */
template <class TShape, typename std::enable_if<!std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_labels(const Node* const, TShape&) {}

}  // namespace reshape

namespace v1 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Reshape* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    using namespace ov::util;
    using TDim = typename T::value_type;

    const auto& input_shape = input_shapes[0];
    const auto& pattern_shape = input_shapes[1];
    const auto input_rank = input_shape.rank();
    const auto pattern_shape_rank = pattern_shape.rank();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           pattern_shape_rank.compatible(0) || pattern_shape_rank.compatible(1),
                           "Pattern shape must have rank 1 or be empty");

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    if (const auto output_bounds = get_input_bounds<TRShape, int64_t>(op, 1, ta)) {
        auto pattern_and_minus_one_idx = reshape::get_pattern_and_minus_one_idx<TRShape>(op, *output_bounds);
        auto& output_pattern = pattern_and_minus_one_idx.first;
        const auto minus_one_idx = pattern_and_minus_one_idx.second;

        reshape::set_pattern_labels(op, output_pattern);

        if (pattern_shape_rank.is_static() && pattern_shape_rank.get_length() == 0) {
            NODE_VALIDATION_CHECK(op,
                                  output_pattern[0] == 1,
                                  "The value of scalar shape pattern should be equal to 1!");
            output_pattern.resize(0);
        }

        const auto special_zero = op->get_special_zero();
        TDim output_product{1};

        for (size_t i = 0; i < output_pattern.size(); ++i) {
            const auto& pattern_dim = output_pattern[i];
            if (static_cast<int64_t>(i) == minus_one_idx) {
                output_shape.emplace_back();
            } else if (pattern_dim == 0 && special_zero) {
                if (input_rank.is_dynamic()) {
                    output_shape.emplace_back(dim::inf_bound);
                    output_product = dim::inf_bound;
                } else {
                    NODE_SHAPE_INFER_CHECK(op, input_shapes, i < input_shape.size(), "'0' dimension is out of range");
                    output_shape.push_back(input_shape[i]);
                    // we do not include dimension to output product here and won't include in input
                    // product later because we will divide output_product by input_product. This
                    // dimension contributes to both products equally
                }
            } else {
                output_shape.emplace_back(pattern_dim);
                output_product *= pattern_dim;
            }
        }

        const auto input_product = input_rank.is_static()
                                       ? reshape::resolve_input_product(input_shape, output_pattern, special_zero)
                                       : reshape::Product<TDim>{TDim(dim::inf_bound), TDim(dim::inf_bound)};

        // resolving -1 masked dimension
        const auto has_minus_one_idx = !dim::is_inf_bound(minus_one_idx);
        if (has_minus_one_idx) {
            if (output_product == 0) {
                NODE_VALIDATION_CHECK(op,
                                      input_product.total() == 0,
                                      "Cannot infer '-1' dimension with zero-size output dimension unless at least one "
                                      "input dimension is also zero-size");
                output_shape[minus_one_idx] = 0;
            } else {
                output_shape[minus_one_idx] = reshape::resolve_minus_one_dim(input_product, output_product);
                NODE_VALIDATION_CHECK(op,
                                      !dim::is_empty(output_shape[minus_one_idx]),
                                      "Non-'-1' output dimensions do not evenly divide the input dimensions");
            }
        }

        if (input_shape.is_static() && output_shape.is_static()) {
            const auto zero_dims = std::any_of(output_pattern.begin(), output_pattern.end(), cmp::Equal<TDim>(0));
            const auto backward_compatible_check = (zero_dims && special_zero) || has_minus_one_idx;
            const auto in_out_elements_equal = (input_product.total() == output_product);

            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   backward_compatible_check || in_out_elements_equal,
                                   "Requested output shape ",
                                   output_shape,
                                   " is incompatible with input shape");
        }
    } else if (pattern_shape_rank.is_static()) {
        auto out_rank = pattern_shape_rank.get_length() == 0
                            ? Rank(0)
                            : Rank(pattern_shape[0].get_min_length(), pattern_shape[0].get_max_length());
        output_shape = PartialShape::dynamic(out_rank);
    } else {
        output_shape = PartialShape::dynamic();
    }
    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
