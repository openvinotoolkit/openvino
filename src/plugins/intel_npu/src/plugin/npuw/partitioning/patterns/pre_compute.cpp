// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pre_compute.hpp"

#include <algorithm>

#include "../../logging.hpp"
#include "../../orc.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

namespace opp = ov::pass::pattern;
namespace pre_compute = ov::npuw::patterns::pre_compute;

namespace {
// TODO: copied from common tests
// Writes `rows` rows of a rotate_half-style sin/cos LUT, row i holding the
// coefficients for absolute position i.
// When duplicate=true (LLama2-style rotate_half), the row is inv_freq_size*2 wide and
// its second half mirrors the first (torch.cat([freqs, freqs], dim=-1)). When
// duplicate=false (GPT-style) the row is inv_freq_size wide with no mirroring.
//
// Coefficients are computed in f32 and stored as f16 - the precision the graph-side
// RoPE cache has always used.
static void writeCosSinRows(const std::vector<float>& inverse_freq_fp32,
                            bool duplicate,
                            size_t rows,
                            ov::float16* pcos,
                            ov::float16* psin) {
    const size_t inv_freq_size = inverse_freq_fp32.size();
    const size_t row_width = duplicate ? inv_freq_size * 2 : inv_freq_size;
    std::fill_n(pcos, rows * row_width, ov::float16{0.0f});
    std::fill_n(psin, rows * row_width, ov::float16{0.0f});

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t k = 0; k < inv_freq_size; k++) {
        auto xita_i = inverse_freq_fp32[k];
        ov::float16* row_sin = psin;
        ov::float16* row_cos = pcos;
        for (size_t m = 0; m < rows; m++, row_sin += row_width, row_cos += row_width) {
            row_cos[k] = ov::float16{std::cos(xita_i * static_cast<float>(m))};
            row_sin[k] = ov::float16{std::sin(xita_i * static_cast<float>(m))};
            if (duplicate) {
                row_cos[k + inv_freq_size] = row_cos[k];
                row_sin[k + inv_freq_size] = row_sin[k];
            }
        }
    }
}

// Builds a [1, max_position_embeddings, rotary_ndims] cos/sin LUT pair as graph
// Constants. Used by every RoPE flavour except LongRoPE, which gets host-fed model
// inputs instead.
static ov::OutputVector makeCosSinCache(const size_t max_position_embeddings,
                                        const std::shared_ptr<ov::Node> inverse_frequencies,
                                        bool duplicate = true) {
    const auto inverse_freq_fp32 = ov::as_type_ptr<ov::op::v0::Constant>(inverse_frequencies)->cast_vector<float>();
    const size_t rotary_ndims = duplicate ? inverse_freq_fp32.size() * 2 : inverse_freq_fp32.size();

    const ov::Shape table_shape{1, max_position_embeddings, rotary_ndims};
    ov::Tensor lut_cos(ov::element::f16, table_shape);
    ov::Tensor lut_sin(ov::element::f16, table_shape);
    writeCosSinRows(inverse_freq_fp32,
                    duplicate,
                    max_position_embeddings,
                    lut_cos.data<ov::float16>(),
                    lut_sin.data<ov::float16>());

    auto Cos = std::make_shared<ov::op::v0::Constant>(lut_cos);
    auto Sin = std::make_shared<ov::op::v0::Constant>(lut_sin);

    return {Cos, Sin};
}

static ov::NodeVector calculate_freq(const std::shared_ptr<ov::Node> short_factor_node,
                                     const std::shared_ptr<ov::Node> long_factor_node,
                                     const std::shared_ptr<ov::Node> multiply_node,
                                     const std::shared_ptr<ov::Node> power_node) {
    const auto short_factor = ov::as_type_ptr<ov::op::v0::Constant>(short_factor_node)->cast_vector<float>();
    const auto long_factor = ov::as_type_ptr<ov::op::v0::Constant>(long_factor_node)->cast_vector<float>();
    const auto multiply_const = ov::as_type_ptr<ov::op::v0::Constant>(multiply_node)->cast_vector<float>();
    const auto power_const = ov::as_type_ptr<ov::op::v0::Constant>(power_node)->cast_vector<float>();
    auto factor_size = short_factor.size();

    OPENVINO_ASSERT(short_factor.size() == multiply_const.size() && long_factor.size() == multiply_const.size(),
                    "Invalid constants for LongRopePatternPhi_v5, expected same size for short_factor, long_factor "
                    "and multiply_const");
    OPENVINO_ASSERT(power_const.size() == 1,
                    "Invalid constants for LongRopePatternPhi_v5, expected single value for power_const");

    std::vector<float> freq(factor_size, 0.0f);
    std::vector<float> freq_long(factor_size, 0.0f);
    for (size_t i = 0; i < factor_size; i++) {
        freq[i] = std::pow(short_factor[i] * multiply_const[i], power_const[0]);
        freq_long[i] = std::pow(long_factor[i] * multiply_const[i], power_const[0]);
    }

    auto inv_freq = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq);
    auto inv_freq_long = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq_long);

    return {inv_freq, inv_freq_long};
}

void replaceSinCosByCache(int max_prompt_len, const ov::OutputVector& cache, const pre_compute::RopePatternDesc* rpe) {
    auto inv_freq_size = ov::shape_size(rpe->matched_inv_freq->get_shape());

    LOG_VERB("Making sin-cos cache of size: " << max_prompt_len << "x" << inv_freq_size);

    // Step 1: Define axis (gather along axis 1)
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    // Step 2: Apply Gather for cos and sin
    auto gather_cos = std::make_shared<ov::op::v8::Gather>(cache[0], rpe->matched_position_ids, axis);
    auto gather_sin = std::make_shared<ov::op::v8::Gather>(cache[1], rpe->matched_position_ids, axis);
    LOG_VERB("Created gather op facilitate LUT search: " << gather_cos->get_name() << ", " << gather_cos->get_shape());

    // Step 2: convert fp16->fp32
    auto cos_fp32 = std::make_shared<ov::op::v0::Convert>(gather_cos, ov::element::f32);
    auto sin_fp32 = std::make_shared<ov::op::v0::Convert>(gather_sin, ov::element::f32);

    // Create the squeeze operation required after gather
    auto squeeze_cos = std::make_shared<ov::op::v0::Squeeze>(cos_fp32, axis);
    auto squeeze_sin = std::make_shared<ov::op::v0::Squeeze>(sin_fp32, axis);

    LOG_VERB("Created squeeze_cos op to reduce axis=1: " << squeeze_cos->get_name() << ", "
                                                         << squeeze_cos->get_shape());
    LOG_VERB("Created squeeze_sin op to reduce axis=1: " << squeeze_sin->get_name() << ", "
                                                         << squeeze_sin->get_shape());

    LOG_VERB("Rope cos detected at: " << rpe->matched_cos->get_name() << ", replacing by cache node: "
                                      << gather_cos->get_name() << ", " << gather_cos->get_shape());
    LOG_VERB("Rope sin detected at: " << rpe->matched_sin->get_name() << ", replacing by cache node: "
                                      << gather_sin->get_name() << ", " << gather_sin->get_shape());

    // replacing sin with gather op
    ov::replace_node(rpe->matched_cos, squeeze_cos);
    ov::replace_node(rpe->matched_sin, squeeze_sin);

    // disconnecting gather from rest or subgraph started from concat_1 - only present when the
    // Broadcast node (and its ShapeOf->Gather->Concat shape-compute chain) actually matched.
    if (rpe->matched_concat) {
        auto gather_input_to_concat = rpe->matched_concat->input(0);
        gather_input_to_concat.get_source_output().remove_target_input(gather_input_to_concat);
    }
}

// --- M-RoPE section discovery -----------------------------------------------------------
//
// Unwraps a chain of single-input Reshape nodes (pure flatten/unflatten boilerplate used by the
// traced "SliceAssign"/"GatherAssign" in-place-update idiom), returning the first non-Reshape
// producer found.
std::shared_ptr<ov::Node> skip_reshapes(std::shared_ptr<ov::Node> n) {
    while (n && ov::is_type<ov::op::v1::Reshape>(n)) {
        n = n->input_value(0).get_node_shared_ptr();
    }
    return n;
}

// Reads the single integer value out of a scalar/1-element integer Constant, or nullopt if
// `node` isn't one.
std::optional<int64_t> read_scalar_index(const std::shared_ptr<ov::Node>& node) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!c) {
        return std::nullopt;
    }
    auto values = c->cast_vector<int64_t>();
    if (values.size() != 1) {
        return std::nullopt;
    }
    return values.front();
}

// Reads {begin, end, step, count} off a positive-stride, last-axis opset8::Slice, or nullopt if
// `slice_node` doesn't have that shape (non-constant bounds, non-positive step, or an axis other
// than the last). `count` is the number of channels the slice actually selects: begin,
// begin+step, begin+2*step, ..., for `count` elements (step == 1 gives the common contiguous
// case; step > 1 is an interleaved/strided assignment, e.g. Qwen's "interleaved mrope" layout).
struct SliceRange {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t step = 1;
    int64_t count = 0;
};

std::optional<SliceRange> read_last_axis_slice_range(const std::shared_ptr<ov::op::v8::Slice>& slice_node,
                                                     int64_t full_width) {
    auto begin_node = slice_node->input_value(1).get_node_shared_ptr();
    auto end_node = slice_node->input_value(2).get_node_shared_ptr();
    auto step_node = slice_node->input_value(3).get_node_shared_ptr();
    auto begin = read_scalar_index(begin_node);
    auto end = read_scalar_index(end_node);
    auto step = read_scalar_index(step_node);
    if (!begin || !end || !step) {
        return std::nullopt;
    }
    if (step.value() <= 0) {
        return std::nullopt;
    }
    if (slice_node->get_input_size() > 4) {
        auto axis_node = slice_node->input_value(4).get_node_shared_ptr();
        auto axis = read_scalar_index(axis_node);
        const auto last_axis = static_cast<int64_t>(slice_node->get_input_partial_shape(0).rank().get_length()) - 1;
        if (!axis || (axis.value() != -1 && axis.value() != last_axis)) {
            return std::nullopt;
        }
    }
    auto b = begin.value() < 0 ? begin.value() + full_width : begin.value();
    auto e = end.value() < 0 ? end.value() + full_width : std::min(end.value(), full_width);
    if (b < 0 || e > full_width || b >= e) {
        return std::nullopt;
    }
    const auto step_value = step.value();
    const auto count = (e - b + step_value - 1) / step_value;
    return SliceRange{b, e, step_value, count};
}

// Checks that `node` eventually (through any number of Reshape-unwraps and nested
// ScatterNDUpdate "GatherAssign" write-backs - irrelevant here, see RopePatternMRope's class
// comment) traces back to the shared stacked `transpose_output`, directly or via a Gather
// reading it. Used only as a sanity check on a section's own select-Gather input - does not
// itself extract any section information.
bool reaches_transpose_output(const std::shared_ptr<ov::Node>& node,
                              const ov::Output<ov::Node>& transpose_output,
                              int depth = 0) {
    if (depth > 16) {
        return false;  // guard against unexpectedly deep/cyclic chains
    }
    auto cur = skip_reshapes(node);
    if (!cur) {
        return false;
    }
    if (cur->output(0) == transpose_output) {
        return true;
    }
    if (auto gather = ov::as_type_ptr<ov::op::v8::Gather>(cur)) {
        return reaches_transpose_output(gather->input_value(0).get_node_shared_ptr(), transpose_output, depth + 1);
    }
    if (auto scatter = ov::as_type_ptr<ov::op::v3::ScatterNDUpdate>(cur)) {
        return reaches_transpose_output(scatter->input_value(0).get_node_shared_ptr(), transpose_output, depth + 1);
    }
    return false;
}

// Recognizes the shared M-RoPE frequency prologue standing above every section's terminal
// select-Gather:
//   Transpose(MatMul(Broadcast(Convert?(inv_freq), any) | Convert?(inv_freq),
//                    Convert(Unsqueeze(position_ids, any))),
//             any)
// This can't be expressed as part of the same static (opp::) pattern graph as `concat_2` below,
// because the chain between them is a dynamically-sized, recursively-defined merge (see
// discover_mrope_sections()) - it's instead recognized imperatively, once the merge chain has
// been walked all the way back down to its terminal Gather. Returns {inv_freq, position_ids} on
// success, or nullopt if `node` isn't that Transpose.
std::optional<std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>> match_mrope_prefix(
    const std::shared_ptr<ov::Node>& node) {
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node);
    if (!transpose) {
        return std::nullopt;
    }
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
    if (!matmul) {
        return std::nullopt;
    }

    auto freq_side = matmul->input_value(0).get_node_shared_ptr();
    if (ov::is_type<ov::op::v3::Broadcast>(freq_side)) {
        freq_side = freq_side->input_value(0).get_node_shared_ptr();
    }
    if (ov::is_type<ov::op::v0::Convert>(freq_side)) {
        freq_side = freq_side->input_value(0).get_node_shared_ptr();
    }
    auto inv_freq = ov::as_type_ptr<ov::op::v0::Constant>(freq_side);
    if (!inv_freq) {
        return std::nullopt;
    }

    auto position_side = matmul->input_value(1).get_node_shared_ptr();
    if (ov::is_type<ov::op::v0::Convert>(position_side)) {
        position_side = position_side->input_value(0).get_node_shared_ptr();
    }
    auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(position_side);
    if (!unsqueeze) {
        return std::nullopt;
    }

    return std::make_pair(std::static_pointer_cast<ov::Node>(inv_freq),
                          unsqueeze->input_value(0).get_node_shared_ptr());
}

// Recognizes one "section merge" step feeding into the final Concat: a ScatterNDUpdate whose
// `updates` operand (port 2) is a (Reshape-wrapped) Gather("select")->Slice->optional
// shape-preserving Broadcast("copy_") chain reading one channel sub-range out of the shared
// stacked frequency tensor, and whose `base` operand (port 0) is either the previous
// (Reshape-wrapped) ScatterNDUpdate in the chain (recursed into), or a terminal
// (Reshape-wrapped) Gather("select") reading the shared M-RoPE prologue's Transpose output
// directly (see match_mrope_prefix()) - the first section, matched here as a full, not-yet-
// sliced row; its effective channel range is whatever is left over after subtracting every
// other discovered section's range (computed by the caller).
//
// Appends one entry per explicitly-sliced section found to `sections` (first/terminal section
// discovered first, since validation happens on the way back up the recursion) and returns the
// terminal Gather("select") node on success, or nullptr if the chain doesn't match the expected
// idiom anywhere along the way.
std::shared_ptr<ov::op::v8::Gather> discover_mrope_sections(const std::shared_ptr<ov::Node>& node,
                                                            int64_t full_width,
                                                            std::vector<pre_compute::MRopeSection>& sections) {
    auto cur = skip_reshapes(node);
    if (!cur) {
        return nullptr;
    }

    if (auto gather = ov::as_type_ptr<ov::op::v8::Gather>(cur)) {
        // Terminal: the first section, reading the shared prologue's Transpose output directly.
        auto prefix_ok = match_mrope_prefix(gather->input_value(0).get_node_shared_ptr()).has_value();
        return prefix_ok ? gather : nullptr;
    }

    auto scatter = ov::as_type_ptr<ov::op::v3::ScatterNDUpdate>(cur);
    if (!scatter) {
        return nullptr;
    }

    auto updates = skip_reshapes(scatter->input_value(2).get_node_shared_ptr());
    if (updates && ov::is_type<ov::op::v3::Broadcast>(updates)) {
        updates = updates->input_value(0).get_node_shared_ptr();
    }
    auto slice = ov::as_type_ptr<ov::op::v8::Slice>(updates);
    if (!slice) {
        return nullptr;
    }
    auto range = read_last_axis_slice_range(slice, full_width);
    auto select = ov::as_type_ptr<ov::op::v8::Gather>(skip_reshapes(slice->input_value(0).get_node_shared_ptr()));
    if (!range || !select) {
        return nullptr;
    }
    auto row_index = read_scalar_index(select->input_value(1).get_node_shared_ptr());
    if (!row_index) {
        return nullptr;
    }

    // Recurse into the base (port 0) first - this both discovers earlier sections and, via the
    // terminal first-section Gather, pins down the shared Transpose node every section's own
    // select-Gather is checked against below.
    auto terminal = discover_mrope_sections(scatter->input_value(0).get_node_shared_ptr(), full_width, sections);
    if (!terminal) {
        return nullptr;
    }
    if (!reaches_transpose_output(select->input_value(0).get_node_shared_ptr(), terminal->input_value(0))) {
        return nullptr;
    }

    sections.push_back({row_index.value(), range->begin, range->step, range->count});
    return terminal;
}

// Builds the M-RoPE replacement: gathers the implicit/base row's own position_ids row in full
// from a *non-duplicated* [1, max_prompt_len, rotary_ndims/2] LUT, then - for every
// explicitly-sliced section - gathers that section's own row too and overwrites its (possibly
// strided/interleaved) channel sub-range on top of the base via ScatterUpdate, before
// reproducing the model's own final torch.cat([row, row], dim=-1) duplication. This mirrors the
// original model's own Gather(select)->Slice->ScatterNDUpdate merge exactly - a simple
// sorted-Slice-then-Concat reassembly of channel ranges cannot represent a channel_step > 1
// (interleaved) section assignment.
void replaceSinCosByMRopeCache(const ov::OutputVector& cache, const pre_compute::RopePatternMRope* rpe) {
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto last_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    // Gathers `row_index`'s own position_ids row out of the LUT, producing the FULL (un-sliced)
    // [batch, seq, rotary_ndims/2] frequency row for that rotary axis.
    auto gather_full_row = [&](const ov::Output<ov::Node>& lut, int64_t row_index) {
        // matched_position_ids is [num_sections, batch, seq]; Gather+Squeeze(axis 0) picks this
        // section's row, giving [batch, seq].
        auto row_index_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {row_index});
        auto position_ids_row = std::make_shared<ov::op::v8::Gather>(rpe->matched_position_ids, row_index_const, axis0);
        auto position_ids_row_sq = std::make_shared<ov::op::v0::Squeeze>(position_ids_row, axis0);

        // Gather that row out of the LUT (shape becomes [1, batch, seq, rotary_ndims/2]) and drop
        // the LUT's own leading batch-1 dim.
        auto gathered = std::make_shared<ov::op::v8::Gather>(lut, position_ids_row_sq, axis1);
        auto gathered_fp32 = std::make_shared<ov::op::v0::Convert>(gathered, ov::element::f32);
        return std::make_shared<ov::op::v0::Squeeze>(gathered_fp32, axis0)->output(0);
    };

    // matched_sections holds only the explicit, Slice-discovered sections (see
    // RopePatternMRope::matched_sections's comment in pre_compute.hpp) - the implicit/base row
    // (matched_base_row_index) is never itself sliced.
    const auto& sections = rpe->matched_sections;

    auto merge_rows = [&](const ov::Output<ov::Node>& lut) {
        ov::Output<ov::Node> merged = gather_full_row(lut, rpe->matched_base_row_index);
        for (const auto& section : sections) {
            auto full_row = gather_full_row(lut, section.position_ids_row_index);

            std::vector<int64_t> channel_indices(static_cast<size_t>(section.channel_count));
            for (int64_t k = 0; k < section.channel_count; ++k) {
                channel_indices[static_cast<size_t>(k)] = section.channel_offset + k * section.channel_step;
            }
            auto indices =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{channel_indices.size()}, channel_indices);

            // Pick out this section's own (possibly strided) channels from its full row, then
            // overwrite the same channel positions in the merged row.
            auto updates = std::make_shared<ov::op::v8::Gather>(full_row, indices, last_axis);
            merged = std::make_shared<ov::op::v3::ScatterUpdate>(merged, indices, updates, last_axis);
        }
        return merged;
    };

    auto merged_cos = merge_rows(cache[0]);
    auto merged_sin = merge_rows(cache[1]);

    auto dup_cos = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{merged_cos, merged_cos}, -1);
    auto dup_sin = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{merged_sin, merged_sin}, -1);

    LOG_VERB("M-RoPE cos detected at: " << rpe->matched_cos->get_name() << ", replacing by cache merge of "
                                        << sections.size() << " sections");

    ov::replace_node(rpe->matched_cos, dup_cos);
    ov::replace_node(rpe->matched_sin, dup_sin);
}

}  // namespace

bool ov::npuw::patterns::pre_compute::RopePatternLLama2::is_valid_match(
    const std::shared_ptr<ov::Node>& sin_input) const {
    // When the Broadcast was bypassed (optional `broadcast` matched directly on inv_freq),
    // matched_inv_freq is used as-is with no reshaping, so any extra non-singleton dim (e.g. a
    // real batch dim) would be silently flattened into the frequency axis by makeCosSinCache,
    // corrupting the LUT's shape. Only a singleton-batch direct constant is supported there.
    if (!matched_concat) {
        const auto& inv_freq_shape = matched_inv_freq->get_shape();
        const auto non_unit_dims = std::count_if(inv_freq_shape.begin(), inv_freq_shape.end(), [](size_t dim) {
            return dim != 1;
        });
        if (non_unit_dims > 1) {
            return false;
        }
    }

    // concat_2 matches any Concat following Transpose (2nd input unconstrained), but only
    // torch.cat([freqs, freqs], dim=-1) - both inputs being the exact same tensor - actually
    // duplicates the frequencies. Anything else (e.g. padding with zeros for partial-rotary
    // models) must not be treated as duplication, or the cached LUT's second half would be wrong.
    if (duplicate_freqs && sin_input->input_value(0) != sin_input->input_value(1)) {
        return false;
    }

    return true;
}

ov::npuw::patterns::pre_compute::RopePatternLLama2::RopePatternLLama2() : matcher("sin-cos-matcher") {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto inv_freq = opp::wrap_type<ov::op::v0::Constant>();
    auto inv_freq_convert = opp::optional<ov::op::v0::Convert>({inv_freq->output(0)});
    // Normally inv_freq is dynamically broadcast to the position_ids-derived shape via a
    // ShapeOf->Gather->Concat->Broadcast chain. Some models (e.g. Qwen3) export this already
    // folded into a plain (already-broadcast-shaped) Constant/Convert with no live Broadcast op -
    // make the Broadcast optional so the matcher still finds the pattern in that case (falls
    // back to matching inv_freq_convert's output directly, see Optional's "match by 1st input").
    auto broadcast = opp::optional<ov::op::v3::Broadcast>({inv_freq_convert, concat_1});
    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    // Optional Concat between Transpose and Sin/Cos: present in LLama2-style RoPE
    // (torch.cat([freqs, freqs], dim=-1)), absent in GPT-style RoPE (inv_freq shape [1, n, 1]).
    auto concat_2 = opp::optional<ov::op::v0::Concat>({transpose->output(0), opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    bind_match = [=](const auto& map_sin, const auto& map_cos) {
        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        // concat_1 only participates in the match when the Broadcast node is actually present -
        // when it was folded away (see comment above `broadcast`), concat_1 is left unmatched.
        this->matched_concat.reset();
        if (const auto it = map_sin.find(concat_1); it != map_sin.end()) {
            this->matched_concat = it->second.get_node_shared_ptr();
        }
        this->matched_inv_freq = map_sin.at(inv_freq).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        // Determine if freq duplication was applied by inspecting the actual graph:
        // sin's direct input is Concat (LLama2-style) or Transpose (GPT-style, no dup).
        auto sin_input = this->matched_sin->input_value(0).get_node_shared_ptr();
        this->duplicate_freqs = ov::is_type<ov::op::v0::Concat>(sin_input);

        // The optional nodes above (`broadcast`, `concat_2`) make the pattern match graph shapes
        // it can't actually handle correctly; reject those rather than silently miscomputing the
        // LUT. is_valid_match() below holds both checks together to keep the vetoes in one place.
        if (!is_valid_match(sin_input)) {
            this->accept_match = false;
            return;
        }

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name()
                                     << " (duplicate_freqs=" << duplicate_freqs << ")");
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback(output_sin, output_cos));
}

ov::npuw::patterns::pre_compute::RopePatternMRope::RopePatternMRope() : matcher("mrope-sin-cos-matcher") {
    // Unlike RopePatternLLama2, the shared frequency prologue (inv_freq/position_ids/Broadcast/
    // MatMul/Transpose) can't be wired into the same static pattern graph as `concat_2` below -
    // M-RoPE merges several channel-ranged "sections" of that prologue's stacked output through
    // a dynamically-sized, recursively-defined per-section Gather("select")/Slice/
    // ScatterNDUpdate chain first (see discover_mrope_sections()), which isn't expressible as a
    // fixed sequence of op types. It's instead recognized imperatively below, once the merge
    // chain has been walked all the way back down to its terminal Gather (match_mrope_prefix()).
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({opp::any_input(), opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    bind_match = [=](const auto& map_sin, const auto& map_cos) {
        this->matched_concat.reset();
        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();
        auto matched_concat_2 = map_sin.at(concat_2).get_node_shared_ptr();

        // Only torch.cat([row, row], dim=-1) - both Concat inputs being the exact same tensor -
        // is reproduced by replaceSinCosByMRopeCache's own final self-duplicating Concat.
        this->duplicate_freqs = matched_concat_2->input_value(0) == matched_concat_2->input_value(1);
        if (!this->duplicate_freqs) {
            this->accept_match = false;
            return;
        }

        // The pre-duplication channel width is needed up front to validate each section's Slice
        // range (and to compute the implicit first section's complement range below) - read it
        // off the (pre-duplication) Concat input's own shape, since matched_inv_freq isn't known
        // until discovery below succeeds.
        const auto& pre_dup_shape = matched_concat_2->get_input_partial_shape(0);
        if (pre_dup_shape.rank().is_dynamic() || pre_dup_shape[pre_dup_shape.rank().get_length() - 1].is_dynamic()) {
            this->accept_match = false;
            return;
        }
        const auto full_width = pre_dup_shape[pre_dup_shape.rank().get_length() - 1].get_length();

        this->matched_sections.clear();
        auto first_section_gather = discover_mrope_sections(matched_concat_2->input_value(0).get_node_shared_ptr(),
                                                            full_width,
                                                            this->matched_sections);
        // Require at least one explicit section plus the implicit (complement-range) first one -
        // a single, unsliced section with no merging at all is already RopePatternLLama2's job.
        if (!first_section_gather || this->matched_sections.empty()) {
            this->accept_match = false;
            return;
        }
        auto first_row = read_scalar_index(first_section_gather->input_value(1).get_node_shared_ptr());
        auto prefix = match_mrope_prefix(first_section_gather->input_value(0).get_node_shared_ptr());
        if (!first_row || !prefix) {
            this->accept_match = false;
            return;
        }
        this->matched_inv_freq = prefix->first;
        this->matched_position_ids = prefix->second;
        if (static_cast<int64_t>(ov::shape_size(this->matched_inv_freq->get_shape())) != full_width) {
            this->accept_match = false;
            return;
        }

        // Every discovered section's channels must be disjoint and in range; reject anything
        // else rather than risk a silently wrong (overlapping) LUT merge. Channels no explicit
        // section claims simply keep the implicit/base row's own value (see
        // replaceSinCosByMRopeCache()) - their shape doesn't need to be contiguous or even
        // regular, so no further validation is needed for them.
        std::vector<bool> covered(static_cast<size_t>(full_width), false);
        bool overlap_rejected = false;
        for (const auto& section : this->matched_sections) {
            for (int64_t k = 0; k < section.channel_count; ++k) {
                const auto c = section.channel_offset + k * section.channel_step;
                if (c < 0 || c >= full_width || covered[static_cast<size_t>(c)]) {
                    this->accept_match = false;
                    overlap_rejected = true;
                    break;
                }
                covered[static_cast<size_t>(c)] = true;
            }
            if (overlap_rejected) {
                return;
            }
        }
        this->matched_base_row_index = first_row.value();

        LOG_VERB("M-RoPE found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name()
                                       << ", sections=" << this->matched_sections.size());
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback(output_sin, output_cos));
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi::LongRopePatternPhi() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& inv_freq_short,
                                   const std::shared_ptr<ov::Node>& inv_freq_long) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto cond_offset = MakeConstant();
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, cond_offset});
        auto context_limit = MakeConstant();
        // max(position_ids) + 1 <= original_max_position_embeddings
        auto leq = opp::wrap_type<ov::op::v1::LessEqual>({add, context_limit});

        auto inv_freq_short_conv = opp::optional<ov::op::v0::Convert>({inv_freq_short->output(0)});
        auto inv_freq_long_conv = opp::optional<ov::op::v0::Convert>({inv_freq_long->output(0)});

        // max(position_ids) + 1 <= original_max_position_embeddings ? short_factor : long_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({leq, inv_freq_short_conv, inv_freq_long_conv});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({select, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, leq, red_max, context_limit, cond_offset);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto inv_freq_short = MakeConstant();
    auto inv_freq_long = MakeConstant();

    auto select_cond_max_pos_id = make_select_pattern(position_ids, inv_freq_short, inv_freq_long);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);
    auto context_limit = std::get<3>(select_cond_max_pos_id);
    auto cond_offset = std::get<4>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    bind_match = [=](const auto& map_sin, const auto& map_cos) {
        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_inv_freq = map_sin.at(inv_freq_short).get_node_shared_ptr();
        this->matched_inv_freq_long = map_sin.at(inv_freq_long).get_node_shared_ptr();
        this->matched_context_limit = map_sin.at(context_limit).get_node_shared_ptr();
        this->matched_cond_offset = map_sin.at(cond_offset).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback(output_sin, output_cos));
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi_v5::LongRopePatternPhi_v5() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& short_factor,
                                   const std::shared_ptr<ov::Node>& long_factor,
                                   const std::shared_ptr<ov::Node>& multiply_const,
                                   const std::shared_ptr<ov::Node>& power_const) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto cond_offset = MakeConstant();
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, cond_offset});
        auto context_limit = MakeConstant();
        // max(position_ids) + 1 > original_max_position_embeddings
        auto greater = opp::wrap_type<ov::op::v1::Greater>({add, context_limit});

        auto short_factor_conv = opp::optional<ov::op::v0::Convert>({short_factor->output(0)});
        auto long_factor_conv = opp::optional<ov::op::v0::Convert>({long_factor->output(0)});

        // max(position_ids) + 1 > original_max_position_embeddings ? long_factor : short_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({greater, long_factor_conv, short_factor_conv});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({select, multiply_const});
        auto power = opp::wrap_type<ov::op::v1::Power>({multiply, power_const});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({power, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, greater, red_max, context_limit, cond_offset);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto short_factor = MakeConstant();
    auto long_factor = MakeConstant();

    auto multiply_const = MakeConstant();
    auto power_const = MakeConstant();

    auto select_cond_max_pos_id =
        make_select_pattern(position_ids, short_factor, long_factor, multiply_const, power_const);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);
    auto context_limit = std::get<3>(select_cond_max_pos_id);
    auto cond_offset = std::get<4>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    bind_match = [=](const auto& map_sin, const auto& map_cos) {
        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_short_factor = map_sin.at(short_factor).get_node_shared_ptr();
        this->matched_long_factor = map_sin.at(long_factor).get_node_shared_ptr();
        this->matched_context_limit = map_sin.at(context_limit).get_node_shared_ptr();
        this->matched_cond_offset = map_sin.at(cond_offset).get_node_shared_ptr();
        this->matched_multiply_const = map_sin.at(multiply_const).get_node_shared_ptr();
        this->matched_power_const = map_sin.at(power_const).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback(output_sin, output_cos));
}

namespace {
// Reads the original_max_position_embeddings constant captured by a LongRoPE pattern.
uint64_t read_context_limit(const std::shared_ptr<ov::Node>& limit_node) {
    auto limit_const = ov::as_type_ptr<ov::op::v0::Constant>(limit_node);
    OPENVINO_ASSERT(limit_const, "Invalid LongRoPE match, expected constant context limit");

    const auto limit_values = limit_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(limit_values.size() == 1, "Invalid LongRoPE context limit, expected a single scalar value");
    OPENVINO_ASSERT(limit_values.front() >= 0, "Invalid LongRoPE context limit, expected a non-negative value");
    return static_cast<uint64_t>(limit_values.front());
}

// The mode switch is re-evaluated on the host as max(position_ids) >= context_limit,
// which only reproduces the graph's `max(position_ids) + offset` comparison for an
// offset of exactly 1. The matchers accept any constant there, so refuse to rewrite a
// model whose offset differs rather than silently binding the wrong coefficients.
void check_cond_offset(const std::shared_ptr<ov::Node>& offset_node) {
    auto offset_const = ov::as_type_ptr<ov::op::v0::Constant>(offset_node);
    OPENVINO_ASSERT(offset_const, "Invalid LongRoPE match, expected a constant position-id offset");

    const auto offset_values = offset_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(offset_values.size() == 1 && offset_values.front() == 1,
                    "Unsupported LongRoPE model: the short/long factor switch is only supported in its "
                    "max(position_ids) + 1 vs original_max_position_embeddings form");
}

template <typename PatternT>
std::optional<uint64_t> extract_context_limit(const std::shared_ptr<ov::Model>& model) {
    auto long_rope = std::make_shared<PatternT>();
    std::optional<uint64_t> context_limit;
    long_rope->transform_cb = [&]() {
        check_cond_offset(long_rope->matched_cond_offset);
        const auto matched_limit = read_context_limit(long_rope->matched_context_limit);
        if (context_limit.has_value()) {
            OPENVINO_ASSERT(context_limit.value() == matched_limit,
                            "Inconsistent LongRoPE context limits detected in the model");
        } else {
            context_limit = matched_limit;
        }
    };
    long_rope->run_on_model(model);
    return context_limit;
}
}  // namespace

std::optional<uint64_t> ov::npuw::patterns::pre_compute::extract_longrope_context_limit(
    const std::shared_ptr<ov::Model>& model) {
    if (auto limit = extract_context_limit<LongRopePatternPhi_v5>(model)) {
        return limit;
    }
    return extract_context_limit<LongRopePatternPhi>(model);
}

ov::npuw::patterns::pre_compute::RopeCacheMatcher::RopeCacheMatcher(const uint32_t max_prompt_len,
                                                                    const std::shared_ptr<ov::Model>& model,
                                                                    LongRopeCosSin* out_tables) {
    auto rpe = std::make_shared<RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq, rpe->duplicate_freqs);
        replaceSinCosByCache(max_prompt_len, cache, rpe.get());
    };
    rpe->run_on_model(model);

    // M-RoPE (multi-section rotary, e.g. Qwen2-VL-style) models aren't matched by
    // RopePatternLLama2 above - the stacked per-section frequency tensor is merged into a single
    // row before Sin/Cos rather than concatenated with itself directly. See RopePatternMRope.
    auto mrope = std::make_shared<RopePatternMRope>();
    mrope->transform_cb = [&]() {
        auto cache = makeCosSinCache(max_prompt_len, mrope->matched_inv_freq, /*duplicate=*/false);
        replaceSinCosByMRopeCache(cache, mrope.get());
    };
    mrope->run_on_model(model);

    // LongRoPE cos/sin LUTs are model inputs, not Constants: the host picks the
    // short/long factor mode and binds the matching rows, so no in-graph Select - and
    // hence no npuw_longrope_input scalar - is created for either LongRoPE pattern.
    std::shared_ptr<ov::op::v0::Parameter> lr_cos_param;
    std::shared_ptr<ov::op::v0::Parameter> lr_sin_param;

    // Replaces the matched Sin/Cos with a Gather from two freshly created f16 model
    // inputs, and reports their layout plus both inverse-frequency sets to out_tables.
    auto make_lut_inputs = [&](const pre_compute::RopePatternDesc* rpe_desc,
                               const std::vector<float>& inv_freq_short,
                               const std::vector<float>& inv_freq_long) {
        const size_t rotary_ndims = inv_freq_short.size() * 2;
        OPENVINO_ASSERT(inv_freq_short.size() == inv_freq_long.size(),
                        "Invalid LongRoPE match, expected same size for the short and long factors");

        const ov::Shape lut_shape{1, max_prompt_len, rotary_ndims};
        lr_cos_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, lut_shape);
        lr_sin_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, lut_shape);
        lr_cos_param->set_friendly_name(longrope_cos_input);
        lr_sin_param->set_friendly_name(longrope_sin_input);

        replaceSinCosByCache(max_prompt_len, {lr_cos_param, lr_sin_param}, rpe_desc);

        if (out_tables) {
            out_tables->max_len = max_prompt_len;
            out_tables->rotary_ndims = rotary_ndims;
            out_tables->inv_freq_short = inv_freq_short;
            out_tables->inv_freq_long = inv_freq_long;
        }
    };

    auto long_rpe = std::make_shared<LongRopePatternPhi>();

    long_rpe->transform_cb = [&]() {
        if (lr_cos_param) {
            return;  // one cos/sin cache per model
        }
        check_cond_offset(long_rpe->matched_cond_offset);
        // Unlike v5, here the matched constants already are the inverse frequencies.
        make_lut_inputs(long_rpe.get(),
                        ov::as_type_ptr<ov::op::v0::Constant>(long_rpe->matched_inv_freq)->cast_vector<float>(),
                        ov::as_type_ptr<ov::op::v0::Constant>(long_rpe->matched_inv_freq_long)->cast_vector<float>());
    };
    long_rpe->run_on_model(model);

    auto long_rpe_v5 = std::make_shared<LongRopePatternPhi_v5>();

    long_rpe_v5->transform_cb = [&]() {
        if (lr_cos_param) {
            return;  // one cos/sin cache per model
        }
        check_cond_offset(long_rpe_v5->matched_cond_offset);
        auto inv_freq = calculate_freq(long_rpe_v5->matched_short_factor,
                                       long_rpe_v5->matched_long_factor,
                                       long_rpe_v5->matched_multiply_const,
                                       long_rpe_v5->matched_power_const);

        // WA: to get correct sin-cos cache size
        long_rpe_v5->matched_inv_freq = inv_freq[0];
        make_lut_inputs(long_rpe_v5.get(),
                        ov::as_type_ptr<ov::op::v0::Constant>(inv_freq[0])->cast_vector<float>(),
                        ov::as_type_ptr<ov::op::v0::Constant>(inv_freq[1])->cast_vector<float>());
    };
    long_rpe_v5->run_on_model(model);

    if (lr_cos_param) {
        model->add_parameters({lr_cos_param, lr_sin_param});
        for (auto&& input : model->inputs()) {
            if (input.get_node() == lr_cos_param.get()) {
                input.set_names({lr_cos_param->get_friendly_name()});
            } else if (input.get_node() == lr_sin_param.get()) {
                input.set_names({lr_sin_param->get_friendly_name()});
            }
        }
    }
    model->validate_nodes_and_infer_types();
}

ov::npuw::patterns::pre_compute::RopeInverseFreq::RopeInverseFreq(
    ov::npuw::patterns::pre_compute::RopeInverseFreq::Results need_freq_consts,
    const std::shared_ptr<ov::Model>& model) {
    auto rpe = std::make_shared<ov::npuw::patterns::pre_compute::RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        if (auto inverse_freq_constant = ov::as_type_ptr<ov::op::v0::Constant>(rpe->matched_inv_freq)) {
            LOG_VERB("Inverse Frequences Constant found: " << inverse_freq_constant->get_name());
            need_freq_consts.get().push_back(inverse_freq_constant);
            return true;
        }
        return false;  // root hasnt changed
    };
    rpe->run_on_model(model);
}

bool ov::npuw::patterns::pre_compute::RopeCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::patterns::pre_compute::RopeCacheMatcher ropeCache(m_max_prompt_len, model, &m_host_tables);
    return true;
}

void ov::npuw::patterns::pre_compute::LongRopeCosSin::rebuild_tables() {
    cos = {};
    sin = {};
    if (max_len == 0 || rotary_ndims == 0 || inv_freq_short.empty()) {
        return;
    }
    const size_t modes = has_long ? 2u : 1u;
    const ov::Shape shape{1, modes * max_len, rotary_ndims};
    cos = ov::Tensor(ov::element::f16, shape);
    sin = ov::Tensor(ov::element::f16, shape);

    writeCosSinRows(inv_freq_short, true, max_len, cos.data<ov::float16>(), sin.data<ov::float16>());
    if (has_long) {
        writeCosSinRows(inv_freq_long,
                        true,
                        max_len,
                        cos.data<ov::float16>() + max_len * rotary_ndims,
                        sin.data<ov::float16>() + max_len * rotary_ndims);
    }
}

namespace {
ov::Tensor longrope_rows(ov::Tensor& table,
                         size_t max_len,
                         size_t rotary_ndims,
                         bool has_long,
                         size_t lut_len,
                         bool is_long) {
    OPENVINO_ASSERT(table && lut_len <= max_len, "LongRoPE LUT does not cover the requested rows");
    const size_t row_offset = (is_long && has_long) ? max_len : 0u;
    auto* rows = table.data<ov::float16>() + row_offset * rotary_ndims;
    return ov::Tensor(ov::element::f16, ov::Shape{1, lut_len, rotary_ndims}, rows);
}
}  // namespace

ov::Tensor ov::npuw::patterns::pre_compute::LongRopeCosSin::cos_rows(size_t lut_len, bool is_long) {
    return longrope_rows(cos, max_len, rotary_ndims, has_long, lut_len, is_long);
}

ov::Tensor ov::npuw::patterns::pre_compute::LongRopeCosSin::sin_rows(size_t lut_len, bool is_long) {
    return longrope_rows(sin, max_len, rotary_ndims, has_long, lut_len, is_long);
}

void ov::npuw::patterns::pre_compute::LongRopeCosSin::serialize(ov::npuw::orc::Stream& stream) {
    stream & max_len & rotary_ndims & has_long & inv_freq_short & inv_freq_long;
    if (stream.input()) {
        // Deserialization imports already-compiled child models, so RopeCache never runs
        // again - rebuild the tables here, otherwise the imported graph's npuw_lr_cos/sin
        // inputs would have nothing to bind to.
        rebuild_tables();
    }
}
