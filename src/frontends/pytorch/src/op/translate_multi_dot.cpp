#include <vector>
#include <memory>
#include <limits>
#include <cstdint>
#include <utility>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

// Try to include PT-FE utils if available
#if __has_include("openvino/frontend/pytorch/utils.hpp")
  #include "openvino/frontend/pytorch/utils.hpp"
  #define OV_PTFE_HAVE_UTILS 1
#else
  #define OV_PTFE_HAVE_UTILS 0
#endif

using namespace ov;
namespace ofep = ov::frontend::pytorch;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// ----------------------------- helpers --------------------------------------
//  helper to create & mark nodes
template <class T, class... Args>
static std::shared_ptr<Node> mk(const ofep::NodeContext& ctx, Args&&... args) {
    auto n = std::make_shared<T>(std::forward<Args>(args)...);
    return ctx.mark_node(n);
}

//  rank or -1 if dynamic
static int64_t rank_of(const Output<Node>& x) {
    auto ps = x.get_partial_shape();
    if (!ps.rank().is_static()) return -1;
    return ps.rank().get_length();
}

//  static 2D shape, or empty if rank/dims dynamic
static std::vector<int64_t> static_shape_2d(const Output<Node>& x) {
    std::vector<int64_t> s;
    auto ps = x.get_partial_shape();
    if (!ps.rank().is_static() || ps.rank().get_length() != 2) return s;
    for (int i = 0; i < 2; ++i) {
        if (!ps[i].is_static()) return {};
        s.push_back(static_cast<int64_t>(ps[i].get_length()));
    }
    return s;
}

// unsqueeze 1D vector to 2D row/col
static std::shared_ptr<Node> unsqueeze_if_1d(const ofep::NodeContext& ctx,
                                             const Output<Node>& x, int64_t axis) {
    auto r = rank_of(x);
    if (r == 1) {
        auto ax = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{axis});
        return mk<ov::op::v0::Unsqueeze>(ctx, x, ax);
    }
    return x.get_node_shared_ptr();
}

// squeeze back to 1D if exactly one of endpoints was 1D
static std::shared_ptr<Node> maybe_squeeze_back_to_1d(const ofep::NodeContext& ctx,
                                                      const Output<Node>& x,
                                                      bool first_was_1d,
                                                      bool last_was_1d) {
    if (!(first_was_1d ^ last_was_1d)) return x.get_node_shared_ptr();
    auto ps = x.get_partial_shape();
    if (ps.rank().is_static() && ps.rank().get_length() == 2) {
        int64_t axis = first_was_1d ? 0 : 1;
        auto ax = mk<ov::op::v0::Constant>(ctx, element::i64, Shape{1}, std::vector<int64_t>{axis});
        return mk<ov::op::v0::Squeeze>(ctx, x, ax);
    }
    return x.get_node_shared_ptr();
}

//  try to extract list elements if input is a list; return true if succeeded
static bool try_dequeue_list(const ofep::NodeContext& ctx,
                             const Output<Node>& in,
                             std::vector<Output<Node>>& out) {
#if OV_PTFE_HAVE_UTILS
    // Variant 1: function in root pytorch namespace
    try {
        auto deq = ov::frontend::pytorch::get_list_as_outputs(in, /*unsqueeze_for_concat=*/false);
        if (!deq.empty()) {
            out.insert(out.end(), deq.begin(), deq.end());
            return true;
        }
    } catch (...) { /* fallthrough */ }
    // Variant 2: function under ::utils::
    try {
        using ov::frontend::pytorch::utils::get_list_as_outputs;
        auto deq = get_list_as_outputs(in, /*unsqueeze_for_concat=*/false);
        if (!deq.empty()) {
            out.insert(out.end(), deq.begin(), deq.end());
            return true;
        }
    } catch (...) { /* fallthrough */ }
#endif
    return false;
}

// collect inputs – (A) single-list port or (B) positional ports
static std::vector<Output<Node>> collect_inputs(const ofep::NodeContext& ctx) {
    std::vector<Output<Node>> xs;
    if (ctx.get_input_size() == 1) {
        const auto in0 = ctx.get_input(0);
        if (try_dequeue_list(ctx, in0, xs)) {
            return xs; // list unpacked successfully
        }
        // treat as a single tensor if not a list / utils not available
        xs.push_back(in0);
        return xs;
    }
    for (size_t i = 0; i < ctx.get_input_size(); ++i)
        xs.push_back(ctx.get_input(i));
    return xs;
}

// DP for matrix-chain order. dims[i] = (rows_i, cols_i). Returns split table s.
static bool mco_dp_from_dims(const std::vector<std::pair<int64_t,int64_t>>& dims,
                             std::vector<std::vector<int64_t>>& s_out) {
    const int n = (int)dims.size();
    if (n < 2) return false;

    // Build p[] of length n+1 where dims[i] = (p[i], p[i+1])
    std::vector<int64_t> p(n+1);
    p[0] = dims[0].first;
    for (int i = 0; i < n; ++i) {
        if (dims[i].first <= 0 || dims[i].second <= 0) return false;
        if (i > 0 && dims[i-1].second != dims[i].first) return false; // inner mismatch
        p[i+1] = dims[i].second;
    }

    std::vector<std::vector<long double>> m(n, std::vector<long double>(n, 0));
    std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n, -1));

    for (int L = 2; L <= n; ++L) {
        for (int i = 0; i <= n - L; ++i) {
            int j = i + L - 1;
            m[i][j] = std::numeric_limits<long double>::infinity();
            for (int k = i; k < j; ++k) {
                long double cost = m[i][k] + m[k+1][j] + (long double)p[i]*p[k+1]*p[j+1];
                if (cost < m[i][j]) {
                    m[i][j] = cost;
                    s[i][j] = k;
                }
            }
        }
    }
    s_out = std::move(s);
    return true;
}

// recursively build according to split table
static std::shared_ptr<Node> build_by_split(const ofep::NodeContext& ctx,
 const std::vector<std::shared_ptr<Node>>& mats2d,
 const std::vector<std::vector<int64_t>>& s,
 int i, int j) {
    if (i == j) return mats2d[i];
    int k = s[i][j];
    auto left  = build_by_split(ctx, mats2d, s, i, k);
    auto right = build_by_split(ctx, mats2d, s, k+1, j);
    return mk<ov::op::v0::MatMul>(ctx, left, right, false, false);
}

// left-associative fallback (safe for dynamic shapes)
static std::shared_ptr<Node> build_left_chain(const ofep::NodeContext& ctx,
                                              const std::vector<std::shared_ptr<Node>>& mats2d) {
    auto acc = mats2d[0];
    for (size_t i = 1; i < mats2d.size(); ++i) {
        acc = mk<ov::op::v0::MatMul>(ctx, acc->output(0), mats2d[i]->output(0), false, false);
    }
    return acc;
}

//decide matrix multiplication order and build the chain
static std::shared_ptr<Node> build_multi_dot_chain(const ofep::NodeContext& ctx,
                                                   const std::vector<std::shared_ptr<Node>>& mats2d,
                                                   const std::vector<std::pair<int64_t,int64_t>>& dims,
                                                   bool all_static_2d) {
    const size_t n = mats2d.size();
    FRONT_END_OP_CONVERSION_CHECK(n >= 2,
        "linalg_multi_dot requires at least 2 tensors; got ", n);

    // Case 1: exactly two tensors -> single MatMul (no need for shape heuristics)
    if (n == 2) {
        return mk<ov::op::v0::MatMul>(ctx,
                                      mats2d[0]->output(0),
                                      mats2d[1]->output(0),
                                      false,
                                      false);
    }

    // If shapes are not fully static 2D, use safe left-associative chain
    if (!all_static_2d) {
        return build_left_chain(ctx, mats2d);
    }

    // At this point: all_static_2d == true, dims.size() == n

    // Case 2: exactly three tensors -> PyTorch-like heuristic
    if (n == 3) {
        // Assume A=(a,b), B=(b,c), C=(c,d)
        const int64_t a = dims[0].first;   // rows of A
        const int64_t b = dims[0].second;  // cols of A == rows of B
        const int64_t c = dims[1].second;  // cols of B == rows of C
        const int64_t d = dims[2].second;  // cols of C

        const long long cost_right = (long long)a * c * (b + d); // A @ (B @ C)
        const long long cost_left  = (long long)b * d * (a + c); // (A @ B) @ C

        if (cost_right > cost_left) {
            auto ab = mk<ov::op::v0::MatMul>(ctx,
                                             mats2d[0]->output(0),
                                             mats2d[1]->output(0),
                                             false,
                                             false);
            return mk<ov::op::v0::MatMul>(ctx,
                                          ab->output(0),
                                          mats2d[2]->output(0),
                                          false,
                                          false);
        } else {
            auto bc = mk<ov::op::v0::MatMul>(ctx,
                                             mats2d[1]->output(0),
                                             mats2d[2]->output(0),
                                             false,
                                             false);
            return mk<ov::op::v0::MatMul>(ctx,
                                          mats2d[0]->output(0),
                                          bc->output(0),
                                          false,
                                          false);
        }
    }

    // Case 3: four or more tensors with static 2D shapes -> full DP (Matrix-Chain Order)
    std::vector<std::vector<int64_t>> s;
    bool ok = mco_dp_from_dims(dims, s);
    FRONT_END_OP_CONVERSION_CHECK(ok, "linalg_multi_dot: incompatible static shapes");
    return build_by_split(ctx, mats2d, s, /*i=*/0, /*j=*/static_cast<int>(n) - 1);
}

ov::OutputVector translate_multi_dot(const ofep::NodeContext& ctx) {
    // 1) Gather inputs
    auto ins = collect_inputs(ctx);
    FRONT_END_OP_CONVERSION_CHECK(ins.size() >= 2,
        "linalg_multi_dot requires at least 2 tensors; got ", ins.size());

    // 2) Align dtypes (convert all to first's dtype if needed)
    auto et0 = ins[0].get_element_type();
    for (size_t i = 1; i < ins.size(); ++i) {
        if (ins[i].get_element_type() != et0) {
            ins[i] = mk<ov::op::v0::Convert>(ctx, ins[i], et0);
        }
    }

    // 3) Normalize endpoints 1D->2D (row / col), middles must be 2D or dynamic rank
    const bool first_was_1d = (rank_of(ins.front()) == 1);
    const bool last_was_1d  = (rank_of(ins.back())  == 1);

    std::vector<std::shared_ptr<Node>> mats;
    mats.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        if (i == 0) {
            mats.push_back(unsqueeze_if_1d(ctx, ins[i], /*axis=*/0));
        } else if (i == ins.size() - 1) {
            mats.push_back(unsqueeze_if_1d(ctx, ins[i], /*axis=*/1));
        } else {
            FRONT_END_OP_CONVERSION_CHECK(rank_of(ins[i]) == 2 || rank_of(ins[i]) == -1,
                "linalg_multi_dot expects 2D middle tensors");
            mats.push_back(ins[i].get_node_shared_ptr());
        }
    }

    // 4) Check if all shapes are static 2D and collect dims
    bool all_static_2d = true;
    std::vector<std::pair<int64_t,int64_t>> dims;
    dims.reserve(mats.size());
    for (auto& m : mats) {
        auto s = static_shape_2d(m->output(0));
        if (s.empty()) {
            all_static_2d = false;
            break;
        }
        dims.emplace_back(s[0], s[1]);
    }

    // 5) Delegate chain-building strategy to helper
    std::shared_ptr<Node> prod =
        build_multi_dot_chain(ctx, mats, dims, all_static_2d);

    // 6) squeeze back to 1D if needed (exactly one endpoint was 1D)
    prod = maybe_squeeze_back_to_1d(ctx, prod->output(0), first_was_1d, last_was_1d);

    return { prod };
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov



