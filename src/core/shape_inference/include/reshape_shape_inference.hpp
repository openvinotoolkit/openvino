// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace reshape {
template <class T, class U = void>
struct Product {};

/** \brief Helper to resolve the input and output product for static dimensions. */
template <class T>
struct Product<T, typename std::enable_if<!std::is_same<T, Dimension>::value>::type> {
    T in{1};
    T out{1};

    void update_in(const T& in_dim) {
        in *= in_dim;
    }

    void update_out(const T& out_dim) {
        out *= out_dim;
    }

    void set_inf() {
        in = T(-1);
        out = T(-1);
    }

    const T& get_static_in() const {
        return in;
    }

    const T& get_static_out() const {
        return out;
    }

    void calculate() {}
};

/** \brief Helper to resolve the input and output product for ov::Dimension (dynamic) dimensions. */
template <class T>
struct Product<T, typename std::enable_if<std::is_same<T, Dimension>::value>::type> {
    std::pair<T, T> in{1, 1};
    std::pair<T, T> out{1, 1};

    void update_in(const T& in_dim) {
        inputs.emplace_back(in_dim);
    }

    void update_out(const T& out_dim) {
        outputs.emplace_back(out_dim);
    }

    void set_inf() {
        in.second = T(-1);
        out.second = T(-1);
    }

    const T& get_static_in() const {
        return in.first;
    }

    const T& get_static_out() const {
        return out.first;
    }

    const T& get_dynamic_in() const {
        return in.second;
    }

    const T& get_dynamic_out() const {
        return out.second;
    }

    void calculate() {
        // dimensions compare to remove same from product calculation
        auto dim_full_eq = [](const T& lhs, const T& rhs) -> bool {
            bool symbols_equal_or_both_null =
                ov::symbol::are_equal(lhs.get_symbol(), rhs.get_symbol()) || (!lhs.has_symbol() && !rhs.has_symbol());
            return (lhs == rhs) && symbols_equal_or_both_null && (lhs.is_static() || lhs.has_symbol());
        };

        auto outs = outputs;

        // calculate input product
        for (const auto& d : inputs) {
            auto out_it = std::find_if(outs.begin(), outs.end(), [&](const T& p) {
                return dim_full_eq(d, p) && (d != 0);
            });

            if (out_it == outs.end()) {
                mul(in, d);
            } else if (!outs.empty()) {
                outs.erase(out_it);
            }
        }

        // calculate output product
        for (const auto& o : outs) {
            mul(out, o);
        }

        if (in.first != out.first) {
            in.second *= in.first;
            out.second *= out.first;
        } else if (in.first == 1 && in.second == 1) {
            // If dynamic product is one (no dynamic) and static is also one use static
            in.second = in.first;
        }
    }

private:
    void mul(std::pair<T, T>& prod, const T& value) {
        if (value.is_static()) {
            prod.first = value * prod.first;
        } else {
            prod.second = value * prod.second;
        }
    }

    std::vector<T> inputs{};
    std::vector<T> outputs{};
};

// resolve minus one dimension for ov::Dimension
template <class TDim,
          typename std::enable_if<std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
TDim resolve_minus_one_dim(const Product<TDim>& product) {
    auto minus_one_dim = product.get_dynamic_in();
    auto& product_out = product.get_dynamic_out();

    if (minus_one_dim.is_static() && product_out.is_static()) {
        minus_one_dim /= product_out.get_length();
    } else {
        using namespace ov::util;
        auto& minus_one_interval = minus_one_dim.get_interval();

        if (minus_one_interval.has_upper_bound() && product_out.get_min_length() != 0 && product_out != TDim{}) {
            minus_one_interval.set_max_val(minus_one_interval.get_max_val() / product_out.get_min_length());
        } else {
            minus_one_interval.set_max_val(Interval::s_max);
        }

        if (product_out.get_max_length() != 0) {
            minus_one_interval.set_min_val(
                ceil_div(minus_one_interval.get_min_val(), product_out.get_interval().get_max_val()));
        }

        if (product_out.get_min_length() != 1 || product_out.get_max_length() != 1) {
            minus_one_dim.set_symbol(nullptr);
        }
    }
    return minus_one_dim;
}

// resolve minus one dimension for static dimension
template <class TDim,
          typename std::enable_if<!std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
TDim resolve_minus_one_dim(const Product<TDim>& product) {
    return product.get_static_in() / product.get_static_out().get_length();
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

        if (*bounds_iter >= minus_one_bound) {
            shape.emplace_back(bounds_iter->first, bounds_iter->second);
        } else if ((bounds_iter->first < 0) != (bounds_iter->second < 0)) {
            // only one bound valid
            shape.emplace_back(0, dim::inf_bound);
        } else {
            NODE_VALIDATION_CHECK(op,
                                  false,
                                  "Output pattern dim[",
                                  i,
                                  "] has invalid bounds: ",
                                  bounds_iter->first,
                                  ",",
                                  bounds_iter->second);
        }
    }

    return result;
}

/**
 * @brief Set the pattern symbols on pattern shape if this input has symbols.
 *
 * @param op     Pointer to reshape node.
 * @param shape  Pointer to shape for symbols set.
 */
template <class TShape, typename std::enable_if<std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_symbols(const Node* const op, TShape& shape) {
    if (op->get_input_size() > 0) {
        auto symbols = op->get_input_source_output(1).get_tensor().get_value_symbol();

        if (!symbols.empty()) {
            auto symbol_iter = symbols.begin();
            for (auto& d : shape) {
                d.set_symbol(*symbol_iter);
                ++symbol_iter;
            }
        }
    }
}

/** @brief Shapes other than PartialShape have no symbols. */
template <class TShape, typename std::enable_if<!std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_symbols(const Node* const, TShape&) {}

/** @brief Deducing symbol relations: number of elements in the tensor doesn't change after the Reshape operation. */
template <class TDim,
          typename std::enable_if<std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
void deduce_symbol_relations(const Product<TDim>& product) {
    auto dyn_in = product.get_dynamic_in();
    auto dyn_out = product.get_dynamic_out();
    dyn_in.merge(dyn_in, dyn_in, dyn_out);
}

/** @brief Shapes other than PartialShape have no symbols. */
template <class TDim,
          typename std::enable_if<!std::is_same<typename std::decay<TDim>::type, Dimension>::value>::type* = nullptr>
void deduce_symbol_relations(const Product<TDim>& product) {}

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

        reshape::set_pattern_symbols(op, output_pattern);

        if (pattern_shape_rank.get_max_length() == 0) {
            NODE_VALIDATION_CHECK(op,
                                  output_pattern[0] == 1,
                                  "The value of scalar shape pattern should be equal to 1!");
            output_pattern.resize(0);
        }

        const auto special_zero = op->get_special_zero();

        reshape::Product<TDim> product;

        if (input_rank.is_dynamic()) {
            for (const auto& pattern : output_pattern) {
                if (special_zero && pattern == 0) {
                    output_shape.emplace_back(dim::inf_bound);
                    product.set_inf();
                } else {
                    output_shape.emplace_back(pattern);
                    product.update_out(pattern);
                }
            }
        } else {
            auto input_iter = input_shape.begin();
            auto input_last = input_shape.end();

            for (size_t i = 0; i < output_pattern.size(); ++i) {
                const auto& pattern_dim = output_pattern[i];
                auto ignore_pattern_dim = special_zero && (pattern_dim == 0);

                if (static_cast<int64_t>(i) == minus_one_idx) {
                    output_shape.emplace_back();
                } else if (ignore_pattern_dim) {
                    NODE_SHAPE_INFER_CHECK(op, input_shapes, i < input_shape.size(), "'0' dimension is out of range");
                    output_shape.push_back(*input_iter);
                    // Exclude special zero dimension from product calculation
                } else {
                    output_shape.push_back(pattern_dim);
                    product.update_out(pattern_dim);
                }

                if (input_iter != input_last) {
                    if (!ignore_pattern_dim) {
                        product.update_in(*input_iter);
                    }
                    ++input_iter;
                }
            }

            // update input product by remaining input dimensions.
            for (; input_iter != input_last; ++input_iter) {
                product.update_in(*input_iter);
            }
        }
        product.calculate();

        // resolving -1 masked dimension
        const auto has_minus_one_idx = !dim::is_inf_bound(minus_one_idx);
        if (has_minus_one_idx) {
            auto& minus_one_dim = output_shape[minus_one_idx];
            minus_one_dim = reshape::resolve_minus_one_dim(product);

            if (product.get_static_out() == 0) {
                NODE_VALIDATION_CHECK(op,
                                      product.get_static_in() == 0,
                                      "Cannot infer '-1' dimension with zero-size output dimension unless at least one "
                                      "input dimension is also zero-size");
            } else {
                NODE_VALIDATION_CHECK(op,
                                      !dim::is_empty(minus_one_dim),
                                      "Non-'-1' output dimensions do not evenly divide the input dimensions");
            }
        } else {
            if (product.get_static_in() == product.get_static_out() && product.get_static_in() != 0) {
                deduce_symbol_relations(product);
            }
        }

        if (input_shape.is_static() && output_shape.is_static()) {
            const auto zero_dims = std::any_of(output_pattern.begin(), output_pattern.end(), cmp::Equal<TDim>(0));
            const auto backward_compatible_check = (zero_dims && special_zero) || has_minus_one_idx;
            const auto in_out_elements_equal = (product.get_static_in() == product.get_static_out());

            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   backward_compatible_check || in_out_elements_equal,
                                   "Requested output shape ",
                                   output_shape,
                                   " is incompatible with input shape");
        }
    } else if (pattern_shape_rank.is_static()) {
        if (pattern_shape_rank.get_length() == 0) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   input_rank.compatible(0),
                                   "Input must be scalar as pattern is scalar!");
        } else {
            output_shape =
                PartialShape::dynamic(Rank(pattern_shape[0].get_min_length(), pattern_shape[0].get_max_length()));
        }
    } else {
        output_shape = PartialShape::dynamic();
    }
    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
