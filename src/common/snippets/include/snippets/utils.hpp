// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public utilities.
 * @file utils.hpp
 */
#pragma once

#include "snippets_isa.hpp"
#include "emitter.hpp"
#include "shape_types.hpp"


namespace ov {
namespace snippets {
namespace utils {

// Get non-scalar Constant count that will be created after FakeQuantize decomposition.
// This count is needed to know exact count of non-scalar Constants during tokenization.
auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ov::opset1::FakeQuantize>& fq) -> size_t;

inline auto is_scalar_constant(const std::shared_ptr<ov::Node>& source_output_node) -> bool {
    return ov::is_type<ov::opset1::Constant>(source_output_node) && ov::shape_size(source_output_node->get_shape()) == 1;
}

inline auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + static_cast<int32_t>(shape_rank) + 1 : allocation_rank;
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename T, typename P>
constexpr bool everyone_is(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

constexpr inline bool implication(bool cause, bool cond) {
    return !cause || !!cond;
}

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    OPENVINO_ASSERT(b != 0, "Divider must not be zero");
    return static_cast<T>((a + b - 1) / b);
}

/* ----- Shape `getters` ----- */
/**
 * @brief Returns a dense shape after applying the order.
 *        It means that the shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param shape preordered (original) partial shape
 * @param order order
 * @return reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [32, 16, 2, 64]
 */
ov::PartialShape get_planar_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order);
/**
 * @brief Returns original shape before applying the order.
 *        It means that the shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param shape planar (ordered) partial shape
 * @param order order
 * @return preordered partial shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [2, 32, 16, 64]
 */
ov::PartialShape get_preordered_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order);
/**
 * @brief Returns a dense shape of node input.
 *        It means that the node input shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param in input of node
 * @return new reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 */
ov::PartialShape get_planar_pshape(const Input<Node>& in);
/**
 * @brief Returns original shape of node output before applying the order.
 *        It means that the preordered output shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param out output of node
 * @return preordered partial shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 */
ov::PartialShape get_preordered_pshape(const Output<Node>& out);
/**
 * @brief Returns a dense shape after applying the order.
 *        It means that the shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param shape preordered (original) shape
 * @param order order
 * @return reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [32, 16, 2, 64]
 */
VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& order);
/**
 * @brief Returns original shape before applying the order.
 *        It means that the preordered shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param shape planar (ordered) shape
 * @param order order
 * @return preordered shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [2, 32, 16, 64]
 */
VectorDims get_preordered_vdims(const VectorDims& shape, const std::vector<size_t>& order);
/**
 * @brief Returns a dense shape of expression input port.
 *        It means that the input shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param expr_port input expression port
 * @return new reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 */
VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port);
/**
 * @brief Returns original shape before applying the order of expression output port.
 *        It means that the preordered output shape dimensions has been already reordered in accordance with order indices to produce planar shape
 * @param out input of node
 * @return preordered shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 */
VectorDims get_preordered_vdims(const snippets::lowered::ExpressionPort& expr_port);

bool is_dynamic_vdims(const VectorDims& shape);

VectorDims pshape_to_vdims(const PartialShape&);
ov::PartialShape vdims_to_pshape(const VectorDims&);
/* --------------------------- */

} // namespace utils
} // namespace snippets
} // namespace ov
