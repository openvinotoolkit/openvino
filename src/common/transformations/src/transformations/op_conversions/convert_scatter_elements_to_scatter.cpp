// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_elements_to_scatter.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/squeeze.hpp"

ov::pass::ConvertScatterElementsToScatter::ConvertScatterElementsToScatter() {
    MATCHER_SCOPE(ConvertScatterElementsToScatter);
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto indices = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto updates = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto axis = ov::op::v0::Constant::create(element::i64, {1}, {0});

    auto broadcast_shape = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(indices, broadcast_shape);

    auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(data, broadcast, updates, axis);

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto scatter = m.get_match_root();
        auto broadcast = scatter->input_value(1).get_node_shared_ptr();
        auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(scatter->input_value(3).get_node_shared_ptr());

        if (!axis_const) {
            return false;
        }

        auto indices_input = broadcast->input_value(0);

        const auto data_pshape = scatter->input(0).get_partial_shape();
        const auto indices_pshape = indices_input.get_partial_shape();
        const auto updates_pshape = scatter->input(2).get_partial_shape();

        // Check that ScatterElementsUpdate and Broadcast inputs has static shapes
        if (data_pshape.rank().is_dynamic() || indices_pshape.rank().is_dynamic() ||
            updates_pshape.rank().is_dynamic()) {
            return false;
        }

        const uint64_t data_rank = data_pshape.rank().get_length();
        const uint64_t updates_rank = updates_pshape.rank().get_length();
        const uint64_t indices_rank = indices_pshape.rank().get_length();

        // Check that axis Constant has {} or {1} shape
        if (shape_size(axis_const->get_shape()) > 1) {
            return false;
        }

        const auto axis =
            ov::util::try_normalize_axis(axis_const->cast_vector<int64_t>()[0], data_pshape.rank(), *scatter);

        struct Range {
            uint64_t l, r;
            Range(const uint64_t& l, const uint64_t& r) : l(l), r(r) {
                if (l > r)
                    OPENVINO_THROW("Range values are inconsistent");
            }

            uint64_t size() const {
                return r - l;
            }

            bool operator!=(const Range& rhs) const {
                return (r - l != rhs.r - rhs.l);
            }

            static bool is_valid(const int64_t& l, const int64_t& r) {
                return (l >= 0 && l <= r);
            }

            static bool is_empty(const uint64_t& l, const uint64_t& r) {
                return l == r;
            }
        };

        auto compare_shapes_ranges = [](const PartialShape& lhsShape,
                                        const PartialShape& rhsShape,
                                        const Range& lhsRange,
                                        const Range& rhsRange) -> bool {
            // Check that ranges are equal and suits to Shapes sizes
            if (lhsRange != rhsRange || lhsRange.r > static_cast<uint64_t>(lhsShape.rank().get_length()) ||
                rhsRange.r > static_cast<uint64_t>(rhsShape.rank().get_length())) {
                return false;
            }

            // Check that Shape values in ranges are equal
            for (size_t lhsIndex = lhsRange.l, rhsIndex = rhsRange.l; lhsIndex < lhsRange.r; ++lhsIndex, ++rhsIndex) {
                if (lhsShape[lhsIndex].is_dynamic() || rhsShape[rhsIndex].is_dynamic() ||
                    lhsShape[lhsIndex] != rhsShape[rhsIndex]) {
                    return false;
                }
            }

            return true;
        };

        auto product = [](const Shape& shape, const Range& range) -> uint64_t {
            uint64_t prod(1);
            for (size_t dim = range.l; dim < range.r; ++dim) {
                prod *= shape[dim];
            }
            return prod;
        };

        /* To transform ScatterElementsUpdate to ScatterUpdate input shapes must match this rules:
         *
         *     data_shape[d_0, d_1, ... , d_n]
         *
         *     indices_shape[i_0, i_1, ... , i_n]
         *
         *     updates_shape[d_0, d_1, i_0(axis), i_1, ... , i_n, d_axis + 1, ... , d_n]
         *
         * EXAMPLE:
         *     In this example the input shapes are suits the rules above and ScatterElementsUpdate can be replaced with
         * ScatterUpdate
         *
         *     axis = 1               | (axis)
         *                           \/
         *
         *     data_shape    [1000, 256,    10, 15]
         *
         *     index_shape   [      125, 2        ]
         *
         *     updates_shape [1000, 125, 2, 10, 15]
         *
         */

        // data_shape and updates_shape dims must be equal up to axis dimension
        if (!compare_shapes_ranges(data_pshape, updates_pshape, {0, axis}, {0, axis})) {
            return false;
        }

        // data_shape dims starting right after axis dim must match last updates_shape dimensions
        if (!Range::is_valid(updates_rank - (data_rank - (axis + 1)), updates_rank)) {
            return false;
        }

        const Range updates_last{updates_rank - (data_rank - (axis + 1)), updates_rank};
        if (!compare_shapes_ranges(data_pshape, updates_pshape, {axis + 1, data_rank}, updates_last)) {
            return false;
        }

        // indices_shape dims product must match updates_shape dims starting from axis dimension
        if (!Range::is_valid(axis, updates_last.l) && !Range::is_empty(axis, updates_last.l)) {
            return false;
        }

        NodeVector new_ops;

        // In case of static shapes we check that indices dims product match with updates dims
        if (updates_pshape.is_static() && indices_pshape.is_static()) {
            const auto updated_range_prod = product(updates_pshape.get_shape(), {axis, updates_last.l});
            const auto indices_range_prod = product(indices_pshape.get_shape(), {0, indices_rank});

            if (updated_range_prod != indices_range_prod) {
                return false;
            }

            // if indices_shape do not match updates_shape dims{axis, updates_last.l}
            // we reshape indices to updates_shape
            const auto updates_shape = updates_pshape.get_shape();
            const auto indices_shape = indices_pshape.get_shape();
            Shape indices_new_shape(updates_shape.begin() + axis, updates_shape.begin() + updates_last.l);
            if (indices_shape != indices_new_shape) {
                indices_input = std::make_shared<ov::op::v1::Reshape>(
                    indices_input,
                    ov::op::v0::Constant::create(element::i64, Shape{indices_new_shape.size()}, indices_new_shape),
                    false);
                new_ops.push_back(indices_input.get_node_shared_ptr());
            }
        } else {
            // Tight constrain for dynamic case:
            // 1. indices_pshape 1...N dimensions must be equal to 1
            // 2. updates_pshape axis interval size = 1

            for (size_t dim = 1; dim < indices_rank; ++dim) {
                if (indices_pshape[dim] != 1)
                    return false;
            }

            if (Range(axis, updates_last.l).size() != 1) {
                return false;
            }

            // Squeeze 1 dims for indices input
            if (indices_rank > 1) {
                std::vector<int64_t> squeeze_axes(indices_rank - 1ul);
                std::iota(squeeze_axes.begin(), squeeze_axes.end(), 1);
                indices_input = std::make_shared<ov::op::v0::Squeeze>(
                    indices_input,
                    ov::op::v0::Constant::create(element::i64, Shape{squeeze_axes.size()}, squeeze_axes));
                new_ops.push_back(indices_input.get_node_shared_ptr());
            }
        }

        auto scatter_update = std::make_shared<ov::op::v3::ScatterUpdate>(scatter->input_value(0),
                                                                          indices_input,
                                                                          scatter->input_value(2),
                                                                          scatter->input_value(3));
        new_ops.push_back(scatter_update);
        scatter_update->set_friendly_name(scatter->get_friendly_name());
        ov::copy_runtime_info({scatter, broadcast}, {new_ops});
        ov::replace_node(scatter, scatter_update);
        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(scatter, matcher_name);
    register_matcher(m, callback);
}
