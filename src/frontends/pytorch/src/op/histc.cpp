// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
constexpr int64_t HISTC_DEFAULT_BINS = 100;

// Tries to constant-fold a scalar input to a double; returns false if it isn't statically known.
bool try_get_scalar_double(const NodeContext& context, size_t index, double& value) {
    if (context.input_is_none(index)) {
        return false;
    }
    const auto const_node = ov::util::get_constant_from_source(context.get_input_from_visible_context(index));
    if (!const_node || shape_size(const_node->get_shape()) == 0) {
        return false;
    }
    value = const_node->cast_vector<double>().at(0);
    return true;
}
}  // namespace

OutputVector translate_histc(const NodeContext& context) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
    // aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
    // torch.export does not materialize defaulted arguments, so bins/min/max may be absent.
    num_inputs_check(context, 1, 5);
    const auto x = context.get_input(0);

    // ATen resolves bin edges in double and computes positions in at::acc_type. Doing the same math
    // in f16 instead overflows to inf as soon as span * bins exceeds 65504, which silently drops
    // samples, so the index math always runs in at least f32. Integer inputs are accepted (ATen
    // supports them on CUDA) and converted like any other dtype.
    const auto compute_type = x.get_element_type() == element::f64 ? element::f64 : element::f32;

    auto flat_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto flat_x = context.mark_node(std::make_shared<v1::Reshape>(x, flat_shape, false));
    auto flat = context.mark_node(std::make_shared<v0::Convert>(flat_x, compute_type));

    // bins is a plain int; constant-fold it for an eager validity check matching ATen's error message,
    // otherwise fall back to a dynamic depth (same tolerance as translate_one_hot's num_classes).
    double bins_value = 0;
    Output<Node> bins_i64;
    if (try_get_scalar_double(context, 1, bins_value)) {
        const auto bins_int = static_cast<int64_t>(bins_value);
        PYTORCH_OP_CONVERSION_CHECK(bins_int > 0, "aten::histc: bins must be greater than 0, but got ", bins_int);
        bins_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {bins_int}));
    } else if (context.input_is_none(1)) {
        bins_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {HISTC_DEFAULT_BINS}));
    } else {
        bins_i64 = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::i64));
    }

    double min_value = 0;
    double max_value = 0;
    const bool min_is_static = context.input_is_none(2) || try_get_scalar_double(context, 2, min_value);
    const bool max_is_static = context.input_is_none(3) || try_get_scalar_double(context, 3, max_value);

    Output<Node> user_min;
    if (context.input_is_none(2)) {
        user_min = context.mark_node(v0::Constant::create(compute_type, Shape{}, {0}));
    } else {
        user_min = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), compute_type));
    }
    Output<Node> user_max;
    if (context.input_is_none(3)) {
        user_max = context.mark_node(v0::Constant::create(compute_type, Shape{}, {0}));
    } else {
        user_max = context.mark_node(std::make_shared<v0::Convert>(context.get_input(3), compute_type));
    }

    Output<Node> left;
    Output<Node> right;
    if (min_is_static && max_is_static && min_value != max_value) {
        // Fast path: fully static range, matching ATen's behavior when min != max is a literal --
        // the data is never inspected and no runtime range-resolution ops are needed.
        PYTORCH_OP_CONVERSION_CHECK(std::isfinite(min_value) && std::isfinite(max_value),
                                    "aten::histc: range of [",
                                    min_value,
                                    ", ",
                                    max_value,
                                    "] is not finite");
        PYTORCH_OP_CONVERSION_CHECK(min_value < max_value, "aten::histc: max must be larger than min");
        left = user_min;
        right = user_max;
    } else {
        // ATen auto-ranges from the data whenever the resolved min == max, then widens a still
        // degenerate range by +/-1. A non-constant min/max forces the same logic to run at runtime.
        // Unlike ATen, a resolved non-finite range cannot be rejected here and yields garbage counts.
        auto eq_user = context.mark_node(std::make_shared<v1::Equal>(user_min, user_max));
        auto reduce_axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
        auto count = numel(context, flat, element::i64);
        auto zero_numel = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
        auto data_nonempty = context.mark_node(std::make_shared<v1::Greater>(count, zero_numel));
        auto use_auto_range = context.mark_node(std::make_shared<v1::LogicalAnd>(eq_user, data_nonempty));
        auto data_min = context.mark_node(std::make_shared<v1::ReduceMin>(flat, reduce_axis, false));
        auto data_max = context.mark_node(std::make_shared<v1::ReduceMax>(flat, reduce_axis, false));
        left = context.mark_node(std::make_shared<v1::Select>(use_auto_range, data_min, user_min));
        right = context.mark_node(std::make_shared<v1::Select>(use_auto_range, data_max, user_max));

        auto one_const = context.mark_node(v0::Constant::create(compute_type, Shape{}, {1}));
        auto degenerate = context.mark_node(std::make_shared<v1::Equal>(left, right));
        auto left_widened = context.mark_node(std::make_shared<v1::Subtract>(left, one_const));
        auto right_widened = context.mark_node(std::make_shared<v1::Add>(right, one_const));
        left = context.mark_node(std::make_shared<v1::Select>(degenerate, left_widened, left));
        right = context.mark_node(std::make_shared<v1::Select>(degenerate, right_widened, right));
    }

    auto span = context.mark_node(std::make_shared<v1::Subtract>(right, left));
    auto ge_min = context.mark_node(std::make_shared<v1::GreaterEqual>(flat, left));
    auto le_max = context.mark_node(std::make_shared<v1::LessEqual>(flat, right));
    // NaN fails both comparisons, so it is dropped automatically -- no separate isnan check needed.
    auto mask = context.mark_node(std::make_shared<v1::LogicalAnd>(ge_min, le_max));
    auto safe_flat = context.mark_node(std::make_shared<v1::Select>(mask, flat, left));

    auto bins_compute = context.mark_node(std::make_shared<v0::Convert>(bins_i64, compute_type));
    auto diff = context.mark_node(std::make_shared<v1::Subtract>(safe_flat, left));
    // exact ATen operand order: (x - left) * bins / span
    auto numer = context.mark_node(std::make_shared<v1::Multiply>(diff, bins_compute));
    auto scaled = context.mark_node(std::make_shared<v1::Divide>(numer, span));
    auto pos_f = context.mark_node(std::make_shared<v0::Floor>(scaled));
    Output<Node> pos = context.mark_node(std::make_shared<v0::Convert>(pos_f, element::i64));

    auto zero_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    // ATen clamps with std::min(bin, bins - 1); x == right would otherwise land one bin past the end.
    auto last_bin = context.mark_node(std::make_shared<v1::Subtract>(bins_i64, one_i64));
    pos = context.mark_node(std::make_shared<v1::Minimum>(pos, last_bin));
    pos = context.mark_node(std::make_shared<v1::Maximum>(pos, zero_i64));

    // Masked-out (NaN or out-of-range) elements are scattered into bin 0 with a zero update.
    auto indices = context.mark_node(std::make_shared<v1::Select>(mask, pos, zero_i64));
    auto zero_count = context.mark_node(v0::Constant::create(compute_type, Shape{}, {0}));
    auto one_count = context.mark_node(v0::Constant::create(compute_type, Shape{}, {1}));
    auto updates = context.mark_node(std::make_shared<v1::Select>(mask, one_count, zero_count));

    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto hist_shape = context.mark_node(std::make_shared<v0::Unsqueeze>(bins_i64, unsqueeze_axis));
    auto histogram = context.mark_node(std::make_shared<v3::Broadcast>(zero_count, hist_shape));
    auto scatter_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto hist =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(histogram,
                                                                       indices,
                                                                       updates,
                                                                       scatter_axis,
                                                                       v12::ScatterElementsUpdate::Reduction::SUM));
    Output<Node> result = context.mark_node(std::make_shared<v1::ConvertLike>(hist, x));

    if (!context.input_is_none(4)) {
        context.mutate_input(4, result);
    }
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
