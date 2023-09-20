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

ov::PartialShape get_planar_pshape(const Input<Node>& out);
ov::PartialShape get_planar_pshape(const Output<Node>& out);
ov::PartialShape get_planar_pshape(const ov::PartialShape& shape, const std::vector<size_t>& layout);
VectorDims pshape_to_vdims(const PartialShape&);
ov::PartialShape vdims_to_pshape(const VectorDims&);

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
    return static_cast<T>((a + b - 1) / b);
}

VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& layout);
VectorDims get_planar_vdims(const snippets::lowered::PortDescriptorPtr& port_desc);
VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port);
bool is_dynamic_vdims(const VectorDims& shape);

} // namespace utils
} // namespace snippets
} // namespace ov
