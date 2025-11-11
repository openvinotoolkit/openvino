// translate_quantile.cpp
#include <vector>
#include <limits>
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::opset11;
using ov::Output;
using ov::OutputVector;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// ---------- helpers ----------

static inline int64_t normalize_dim(int64_t dim, int64_t rank) {
    return dim < 0 ? dim + rank : dim;
}

static std::shared_ptr<ov::Node> axis_const_i64(int64_t v) {
    return Constant::create(element::i64, Shape{}, {v});
}

static std::shared_ptr<ov::Node> gather_axis0(const Output<Node>& data, const Output<Node>& indices) {
    return std::make_shared<Gather>(data, indices, axis_const_i64(0), 0);
}

static std::shared_ptr<ov::Node> gather_dim(const Output<Node>& data, const Output<Node>& indices, int64_t dim) {
    return std::make_shared<Gather>(data, indices, axis_const_i64(dim), 0);
}

// reshape q (rank==1) to [1, ..., len(q), ..., 1] with len(q) at `axis`
static std::shared_ptr<ov::Node>
make_q_fullrank_1d(const ov::Output<ov::Node>& q_f, int64_t self_rank, int64_t axis) {
    auto q_shape    = std::make_shared<ShapeOf>(q_f);
    auto q_len_raw  = gather_axis0(q_shape, Constant::create(element::i64, Shape{1}, {0}));
    auto q_len      = std::make_shared<Squeeze>(q_len_raw);
    auto q_len_unsq = std::make_shared<Unsqueeze>(q_len, Constant::create(element::i64, Shape{1}, {0}));

    const int64_t head = axis;
    const int64_t tail = self_rank - axis - 1;

    auto ones_head = Constant::create(
        element::i64, Shape{static_cast<size_t>(std::max<int64_t>(head, 0))},
        std::vector<int64_t>(std::max<int64_t>(head, 0), 1));

    auto ones_tail = Constant::create(
        element::i64, Shape{static_cast<size_t>(std::max<int64_t>(tail, 0))},
        std::vector<int64_t>(std::max<int64_t>(tail, 0), 1));

    std::vector<Output<Node>> parts;
    if (head > 0) parts.push_back(ones_head);
    parts.push_back(q_len_unsq);
    if (tail > 0) parts.push_back(ones_tail);

    auto pattern = std::make_shared<Concat>(parts, 0);
    return std::make_shared<Reshape>(q_f, pattern, false);
}

static Output<Node> move_dim_to_front(const Output<Node>& node, int64_t src) {
    auto r_ps = node.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(r_ps.is_static(), "translate_quantile: rank must be static for transpose.");
    int64_t r = r_ps.get_length();
    FRONT_END_GENERAL_CHECK(src >= 0 && src < r, "translate_quantile: invalid dimension for transpose.");

    std::vector<int64_t> order;
    order.reserve(static_cast<size_t>(r));
    order.push_back(src);
    for (int64_t i = 0; i < r; ++i)
        if (i != src) order.push_back(i);

    auto perm = Constant::create(element::i64, Shape{static_cast<size_t>(r)}, order);
    return std::make_shared<Transpose>(node, perm);
}

// ---------- main ----------

OutputVector translate_quantile(const NodeContext& context) {
    
    auto self = context.get_input(0);
    auto q    = context.get_input(1);

    // rank BEFORE possible flatten (needed for dim=None + keepdim=True final shape)
    int64_t orig_rank = self.get_partial_shape().rank().is_static()
                        ? self.get_partial_shape().rank().get_length()
                        : 1;

    // PyTorch: dim=None -> flatten, then use dim=0
    bool dim_is_none = context.input_is_none(2);

    int64_t self_rank = self.get_partial_shape().rank().is_static()
                        ? self.get_partial_shape().rank().get_length()
                        : 1;

    if (dim_is_none) {
        auto flat_shape = Constant::create(element::i64, Shape{1}, {-1});
        self = context.mark_node(std::make_shared<Reshape>(self, flat_shape, false));
        self_rank = 1;
    }

    int64_t dim = dim_is_none ? 0 : context.const_input<int64_t>(2);
    dim = normalize_dim(dim, self_rank);

    bool keepdim_req = context.input_is_none(3) ? false : context.const_input<bool>(3);
    bool keepdim_eff = keepdim_req;  // applies even when dim=None (after flatten)

    std::string interpolation = context.input_is_none(4) ? "linear" : context.const_input<std::string>(4);

    // constants
    auto one_f   = Constant::create(element::f32, Shape{}, {1.0f});
    auto zero_f  = Constant::create(element::f32, Shape{}, {0.0f});
    auto half_f  = Constant::create(element::f32, Shape{}, {0.5f});
    auto nan_f   = Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::quiet_NaN()});

    // ---------- NaN mask along reduction dim (keepdim=true semantics for ease of shaping later) ----------
    Output<Node> has_nan_keepdim_true;
    {
        auto isnan = context.mark_node(std::make_shared<IsNaN>(self)); // bool same shape
        if (dim_is_none) {
            // after flatten rank=1 → reduce to scalar
            has_nan_keepdim_true = context.mark_node(
                std::make_shared<ReduceLogicalOr>(isnan, Constant::create(element::i64, Shape{1}, {0}), false));
        } else {
            has_nan_keepdim_true = context.mark_node(
                std::make_shared<ReduceLogicalOr>(isnan, Constant::create(element::i64, Shape{1}, {dim}), true));
        }
    }

    // ---------- sort on `dim` using TopK (K=size(dim)) ----------
    auto self_shape = context.mark_node(std::make_shared<ShapeOf>(self));
    auto k_raw      = context.mark_node(gather_axis0(self_shape, Constant::create(element::i64, Shape{1}, {dim})));
    auto k          = context.mark_node(std::make_shared<Squeeze>(k_raw));

    auto topk = context.mark_node(std::make_shared<TopK>(
        self, k, dim, TopK::Mode::MIN, TopK::SortType::SORT_VALUES, element::i64));
    auto sorted = topk->output(0);

    // n-1 (float scalar)
    auto sorted_shape = context.mark_node(std::make_shared<ShapeOf>(sorted));
    auto n_raw   = context.mark_node(gather_axis0(sorted_shape, Constant::create(element::i64, Shape{1}, {dim})));
    auto n_f32   = context.mark_node(std::make_shared<Convert>(n_raw, element::f32));
    auto last_f1 = context.mark_node(std::make_shared<Subtract>(n_f32, one_f));
    auto last_f  = context.mark_node(std::make_shared<Squeeze>(last_f1));

    // q must be scalar or 1D
    auto q_f = context.mark_node(std::make_shared<Convert>(q, element::f32));
    auto q_rank_ps = q.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(!q_rank_ps.is_static() || q_rank_ps.get_length() <= 1,
        "aten::quantile: only scalar or 1D q is supported");

    // helper: clip indices to [0, n-1]
    auto clip_to_bounds = [&](const Output<Node>& x, const Output<Node>& last_scalar_f) -> Output<Node> {
        auto t1 = context.mark_node(std::make_shared<Minimum>(x, last_scalar_f));
        return context.mark_node(std::make_shared<Maximum>(t1, zero_f));
    };

    bool q_is_scalar = (q_rank_ps.is_static() && q_rank_ps.get_length() == 0);

    if (!q_is_scalar) {
        // ---------- q is 1D ----------
        auto q_fr   = context.mark_node(make_q_fullrank_1d(q_f, self_rank, dim));     // [1,...,M,...,1]
        auto q_rank = context.mark_node(std::make_shared<Multiply>(q_fr, last_f));    // full-rank

        auto floor_f = context.mark_node(std::make_shared<Floor>(q_rank));
        auto ceil_f  = context.mark_node(std::make_shared<Ceiling>(q_rank));

        auto floor_clip_full = clip_to_bounds(floor_f, last_f);
        auto ceil_clip_full  = clip_to_bounds(ceil_f,  last_f);

        auto q_shape2   = std::make_shared<ShapeOf>(q_f);
        auto q_len_raw  = gather_axis0(q_shape2, Constant::create(element::i64, Shape{1}, {0}));
        auto q_len      = std::make_shared<Squeeze>(q_len_raw);
        auto q_len_vec  = std::make_shared<Unsqueeze>(q_len, Constant::create(element::i64, Shape{1}, {0}));

        auto floor_clip = context.mark_node(std::make_shared<Reshape>(floor_clip_full, q_len_vec, false)); // [M]
        auto ceil_clip  = context.mark_node(std::make_shared<Reshape>(ceil_clip_full,  q_len_vec, false)); // [M]

        auto idx_floor = context.mark_node(std::make_shared<Convert>(floor_clip, element::i64));
        auto idx_ceil  = context.mark_node(std::make_shared<Convert>(ceil_clip,  element::i64));

        auto nearest_full      = context.mark_node(std::make_shared<Round>(q_rank, Round::RoundMode::HALF_TO_EVEN));
        auto nearest_clip_full = clip_to_bounds(nearest_full, last_f);
        auto nearest_clip_vec  = context.mark_node(std::make_shared<Reshape>(nearest_clip_full, q_len_vec, false));
        auto idx_near          = context.mark_node(std::make_shared<Convert>(nearest_clip_vec, element::i64));

        Output<Node> res;
        if (interpolation == "lower") {
            res = context.mark_node(gather_dim(sorted, idx_floor, dim));
        } else if (interpolation == "higher") {
            res = context.mark_node(gather_dim(sorted, idx_ceil,  dim));
        } else if (interpolation == "nearest") {
            res = context.mark_node(gather_dim(sorted, idx_near,  dim));
        } else if (interpolation == "midpoint" || interpolation == "linear") {
            auto below = context.mark_node(gather_dim(sorted, idx_floor, dim)); // pre+[M]+post
            auto above = context.mark_node(gather_dim(sorted, idx_ceil,  dim)); // pre+[M]+post
            Output<Node> weight_full = (interpolation == "midpoint")
                ? half_f
                : context.mark_node(std::make_shared<Subtract>(q_rank, floor_f));
            auto one_minus_w = context.mark_node(std::make_shared<Subtract>(one_f, weight_full));
            auto p1 = context.mark_node(std::make_shared<Multiply>(below, one_minus_w));
            auto p2 = context.mark_node(std::make_shared<Multiply>(above, weight_full));
            res = context.mark_node(std::make_shared<Add>(p1, p2));
        } else {
            FRONT_END_THROW("aten::quantile: unsupported interpolation mode: " + interpolation);
        }

        // arrange final output shape before NaN handling
        if (keepdim_eff && !dim_is_none) {
            auto dim_c = Constant::create(element::i64, Shape{1}, {dim});
            res = context.mark_node(std::make_shared<Unsqueeze>(res, dim_c)); // pre+[1]+[M]+post
            res = move_dim_to_front(res, dim + 1);                           // -> [M] + remaining
        } else {
            res = move_dim_to_front(res, dim);                               // -> [M] + remaining
        }

        if (dim_is_none && keepdim_req) {
            auto ones_vec = std::vector<int64_t>(static_cast<size_t>(orig_rank), 1);
            auto ones = Constant::create(element::i64, Shape{ones_vec.size()}, ones_vec);

            auto q_shape3   = std::make_shared<ShapeOf>(q_f);
            auto q_len_raw3 = gather_axis0(q_shape3, Constant::create(element::i64, Shape{1}, {0}));
            auto q_len3     = std::make_shared<Squeeze>(q_len_raw3); // scalar
            auto q_len1     = std::make_shared<Unsqueeze>(q_len3, Constant::create(element::i64, Shape{1}, {0})); // [1]

            std::vector<Output<Node>> parts;
            parts.push_back(q_len1); // [M]
            parts.push_back(ones);   // [1]*orig_rank
            auto tgt = context.mark_node(std::make_shared<Concat>(parts, 0)); // [M, 1, 1, ..., 1]
            res = context.mark_node(std::make_shared<Reshape>(res, tgt, false));
        }

        // ---------- build final-shaped NaN mask and apply ----------
            Output<Node> mask_final;

            if (dim_is_none) {
                
                mask_final = has_nan_keepdim_true;
            } else {
                
                if (keepdim_eff) {
                    
                    auto mask_remain = has_nan_keepdim_true;
                    mask_final = context.mark_node(std::make_shared<Unsqueeze>(
                        mask_remain, Constant::create(element::i64, Shape{1}, {0})));
                } else {
                   
                    auto mask_sq = context.mark_node(std::make_shared<Squeeze>(
                        has_nan_keepdim_true, Constant::create(element::i64, Shape{1}, {dim})));
                    mask_final = context.mark_node(std::make_shared<Unsqueeze>(
                        mask_sq, Constant::create(element::i64, Shape{1}, {0})));
                }
            }

            
        res = context.mark_node(std::make_shared<Select>(mask_final, nan_f, res));

        return {res};

    } else {
        // ---------- q is scalar ----------
        auto q_rank = context.mark_node(std::make_shared<Multiply>(q_f, last_f)); // scalar

        auto floor_f = context.mark_node(std::make_shared<Floor>(q_rank));
        auto ceil_f  = context.mark_node(std::make_shared<Ceiling>(q_rank));

        auto floor_clip = clip_to_bounds(floor_f, last_f);
        auto ceil_clip  = clip_to_bounds(ceil_f,  last_f);

        auto idx_floor = context.mark_node(std::make_shared<Convert>(floor_clip, element::i64)); // scalar idx
        auto idx_ceil  = context.mark_node(std::make_shared<Convert>(ceil_clip,  element::i64)); // scalar idx

        // nearest = round(q_rank) with HALF_TO_EVEN, clipped
        auto nearest     = context.mark_node(std::make_shared<Round>(q_rank, Round::RoundMode::HALF_TO_EVEN));
        auto nearest_clip= clip_to_bounds(nearest, last_f);
        auto idx_near    = context.mark_node(std::make_shared<Convert>(nearest_clip, element::i64));

        Output<Node> result;
        if (interpolation == "lower") {
            result = context.mark_node(gather_dim(sorted, idx_floor, dim));
        } else if (interpolation == "higher") {
            result = context.mark_node(gather_dim(sorted, idx_ceil,  dim));
        } else if (interpolation == "nearest") {
            result = context.mark_node(gather_dim(sorted, idx_near,  dim));
        } else if (interpolation == "midpoint" || interpolation == "linear") {
            auto below = context.mark_node(gather_dim(sorted, idx_floor, dim));
            auto above = context.mark_node(gather_dim(sorted, idx_ceil,  dim));
            Output<Node> weight = (interpolation == "midpoint")
                ? half_f
                : context.mark_node(std::make_shared<Subtract>(q_rank, floor_f));
            auto one_minus_w = context.mark_node(std::make_shared<Subtract>(one_f, weight));
            auto p1 = context.mark_node(std::make_shared<Multiply>(below, one_minus_w));
            auto p2 = context.mark_node(std::make_shared<Multiply>(above, weight));
            result = context.mark_node(std::make_shared<Add>(p1, p2));
        } else {
            FRONT_END_THROW("aten::quantile: unsupported interpolation mode: " + interpolation);
        }

        // arrange final output shape first
        if (keepdim_eff && !dim_is_none) {
            auto dim_c = Constant::create(element::i64, Shape{1}, {dim});
            result = context.mark_node(std::make_shared<Unsqueeze>(result, dim_c));
        }
        if (dim_is_none && keepdim_req) {
            auto ones_vec = std::vector<int64_t>(static_cast<size_t>(orig_rank), 1);
            auto tgt = Constant::create(element::i64, Shape{ones_vec.size()}, ones_vec);
            result = context.mark_node(std::make_shared<Reshape>(result, tgt, false));
        }

        // ---------- build mask EXACTLY matching final result shape ----------
        Output<Node> mask_final;
        if (dim_is_none) {
            // final result is scalar (keepdim=False) or [1]*orig_rank (keepdim=True)
            if (keepdim_req) {
                auto ones_vec = std::vector<int64_t>(static_cast<size_t>(orig_rank), 1);
                auto tgt = Constant::create(element::i64, Shape{ones_vec.size()}, ones_vec);
                mask_final = context.mark_node(std::make_shared<Reshape>(has_nan_keepdim_true, tgt, false)); // [1]*orig_rank
            } else {
                mask_final = has_nan_keepdim_true; // scalar
            }
        } else {
            if (keepdim_eff) {
                mask_final = has_nan_keepdim_true; // has 1 at reduced axis → matches result
            } else {
                mask_final = context.mark_node(std::make_shared<Squeeze>(
                    has_nan_keepdim_true, Constant::create(element::i64, Shape{1}, {dim})));
            }
        }

        // apply NaN mask now (no rank change)
        result = context.mark_node(std::make_shared<Select>(mask_final, nan_f, result));

        return {result};
    }
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
