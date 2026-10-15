// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/adapt_to_genai.hpp"

#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

namespace {

using std::make_shared;

// f16 lowest, matches genai's causal-mask "-inf" fill.
constexpr float NEG_INF = -65504.0f;

// translate_get_rows's embedding-table lookup restores ggml's rank-4 form with a fixed
// Unsqueeze(axis=0), pinning the residual stream's leading axis to a literal 1 regardless of
// backend -- correct for SDPA, but wrong once SDPAToPagedAttention rewrites input_ids to rank-1
// [tokens]: PagedAttentionExtension needs Q/K/V's leading axis to become the real token count, not
// stay 1. That translator is shared with llama.cpp's own raw cgraph decoder (which never runs
// SDPAToPagedAttention and must keep the old, literal-1 layout), so this fix is genai-side only:
// move the restored axis from 0 to 1 for EVERY embedding-table lookup structurally rooted at
// "inp_tokens" -- not just the main "embd" (token_embd.weight) lookup, but also per-layer lookups
// like Gemma4's "pe_tok_flat" (per_layer_token_embd.weight). Matching by name suffix alone (the
// old approach) misses pe_tok_flat: it would then disagree with the fixed embd/projection branch
// on which axis holds the leading placeholder, and their eventual sum broadcasts to a spurious
// [tokens, tokens, ...] shape instead of [tokens, ...]. Axis 1, not 0, because attention's own
// "merge heads back" reshape (op_case 2 in reshape.cpp) already puts a literal 1 there and infers
// axis 2 from whatever is left, so the two must agree on which axis holds the placeholder for
// their sum (ffn_inp) to broadcast correctly; inserting at axis 1 also leaves the pre-PA layout
// numerically unchanged (indices' own leading axis is already 1 there), and still gives Q/K/V's
// leading axis room to become the real token count post-PA.
class FixEmbdAxis : public ov::pass::MatcherPass {
public:
    FixEmbdAxis() {
        using namespace ov::op;
        auto p_unsqueeze = ov::pass::pattern::wrap_type<v0::Unsqueeze>();

        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto unsqueeze = m.get_match_root();

            // translate_get_rows optionally inserts a Convert between the Gather and the
            // Unsqueeze (output dtype mismatch); look through it structurally instead of
            // relying on naming.
            auto node = unsqueeze->input_value(0).get_node_shared_ptr();
            if (auto convert = ov::as_type_ptr<v0::Convert>(node)) {
                node = convert->input_value(0).get_node_shared_ptr();
            }
            auto gather = ov::as_type_ptr<v8::Gather>(node);
            if (!gather) {
                return false;
            }

            // The indices feeding the Gather are Squeeze(inp_tokens, [0,1]) (see
            // translate_get_rows); trace back through that Squeeze to confirm this call site is
            // actually rooted at the "inp_tokens" parameter, not some unrelated Gather/Unsqueeze
            // pair elsewhere in the graph (e.g. FixInpOutIdsRowSelect's call sites).
            auto indices_node = gather->input_value(1).get_node_shared_ptr();
            if (auto squeeze = ov::as_type_ptr<v0::Squeeze>(indices_node)) {
                indices_node = squeeze->input_value(0).get_node_shared_ptr();
            }
            auto param = ov::as_type_ptr<v0::Parameter>(indices_node);
            if (!param || param->output(0).get_names().count("inp_tokens") == 0) {
                return false;
            }

            auto res = unsqueeze->input_value(0);  // Gather's (or Convert's) output, rank 3
            auto axis_1 = v0::Constant::create(ov::element::i64, {1}, {1});
            auto fixed = make_shared<v0::Unsqueeze>(res, axis_1);
            fixed->set_friendly_name(unsqueeze->get_friendly_name());
            ov::copy_runtime_info(unsqueeze, fixed);
            unsqueeze->output(0).replace(fixed->output(0));
            return true;
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(p_unsqueeze, "gguf::FixEmbdAxis"), callback);
    }
};

// inp_out_ids row-selection (attn_out_g/inpSA_g/...) squeezes both leading axes down to a flat
// [tokens, hidden] before gathering, assuming the residual stream's leading axis was always 1;
// now that FixEmbdAxis makes it backend-dependent too, flatten activation and indices to canonical
// [rows, hidden] / [rows, 1] first (Reshape never reorders memory) and gather with a plain
// (non-batched) axis-0 Gather, which is correct either way. Must run before inp_out_ids's own
// value gets replaced (see run_on_model), while it still structurally identifies this call site.
class FixInpOutIdsRowSelect : public ov::pass::MatcherPass {
public:
    FixInpOutIdsRowSelect() {
        using namespace ov::op;
        using ov::pass::pattern::any_input;
        using ov::pass::pattern::wrap_type;

        auto p_inp_out_ids = wrap_type<v0::Parameter>([](const ov::Output<ov::Node>& output) -> bool {
            return output.get_names().count("inp_out_ids") != 0;
        });
        auto p_indices_squeeze = wrap_type<v0::Squeeze>({p_inp_out_ids, any_input()});
        auto p_data_squeeze = wrap_type<v0::Squeeze>({any_input(), any_input()});
        auto p_gather = wrap_type<v8::Gather>({p_data_squeeze, p_indices_squeeze, any_input()});
        auto p_unsqueeze = wrap_type<v0::Unsqueeze>({p_gather, any_input()});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pm = m.get_pattern_value_map();
            auto unsqueeze = m.get_match_root();
            auto data_squeeze = pm.at(p_data_squeeze).get_node_shared_ptr();
            auto indices_squeeze = pm.at(p_indices_squeeze).get_node_shared_ptr();

            auto data = data_squeeze->input_value(0);  // the original, un-squeezed activation
            const int64_t hidden = data.get_partial_shape()[3].get_length();
            auto data_flat_shape = v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, hidden});
            auto data_flat = make_shared<v1::Reshape>(data, data_flat_shape, false);
            auto indices_flat_shape = v0::Constant::create(ov::element::i64, {2}, {-1, 1});
            auto indices_flat = make_shared<v1::Reshape>(indices_squeeze->input_value(0), indices_flat_shape, false);
            auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});
            auto fixed_res = make_shared<v8::Gather>(data_flat, indices_flat, axis0);
            auto fixed_unsqueeze = make_shared<v0::Unsqueeze>(fixed_res, axis0);
            fixed_unsqueeze->set_friendly_name(unsqueeze->get_friendly_name());
            ov::copy_runtime_info(unsqueeze, fixed_unsqueeze);
            unsqueeze->output(0).replace(fixed_unsqueeze->output(0));
            return true;
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(p_unsqueeze, "gguf::FixInpOutIdsRowSelect"),
                         callback);
    }
};

void name_output(const ov::Output<ov::Node>& out, const std::string& name) {
    out.get_node_shared_ptr()->set_friendly_name(name);
    out.get_node_shared_ptr()->output(0).set_names({name});
}

// Largest attention head size across the stateful KV caches (the ReadValue last dim). The
// frontend emits f16 KV caches mirroring llama.cpp, but the CPU plugin defaults
// KV_CACHE_PRECISION to u8 (dynamic-quantized) -- faster and accurate enough for the common
// head sizes used by llama/qwen/phi3/gpt-oss (64-128). For large head sizes the u8
// quantization injects enough per-step error to compound across autoregressive decode into
// divergence and eventually NaN (observed on gemma4, global-attention head_size=512).
int64_t max_kv_cache_head_size(const std::shared_ptr<ov::Model>& model) {
    int64_t max_hs = 0;
    for (const auto& op : model->get_ops()) {
        if (!ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            continue;
        }
        const auto& ps = op->get_output_partial_shape(0);
        if (ps.rank().is_static() && ps[ps.rank().get_length() - 1].is_static()) {
            max_hs = std::max(max_hs, ps[ps.rank().get_length() - 1].get_length());
        }
    }
    return max_hs;
}

}  // namespace

bool AdaptToGenAI::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using namespace ov::op;
    OPENVINO_ASSERT(m_mode == InputMode::IDS_TO_LOGITS,
                    "[gguf] AdaptToGenAI: only InputMode::IDS_TO_LOGITS is implemented; "
                    "EMBEDS_TO_LOGITS (VLM language model) is reserved for future work.");

    // The gguf inputs we rewire. inp_tokens/inp_pos/self_kq_mask/token_len_per_seq are
    // required; if they are absent the model is not a gguf-IO model (e.g. already adapted),
    // so this pass is a no-op.
    auto inp_tokens = find_parameter(model, "inp_tokens");
    auto inp_pos = find_parameter(model, "inp_pos");
    auto self_kq_mask = find_parameter(model, "self_kq_mask");
    auto token_len_per_seq = find_parameter(model, "token_len_per_seq");
    if (!inp_tokens || !inp_pos || !self_kq_mask || !token_len_per_seq) {
        return false;
    }

    // Must run before inp_out_ids's value is replaced below, while these patterns can still
    // structurally identify "embd" and the inp_out_ids-rooted row-selection call sites.
    ov::pass::Manager self_correcting_axis_manager;
    self_correcting_axis_manager.register_pass<FixInpOutIdsRowSelect>();
    self_correcting_axis_manager.register_pass<FixEmbdAxis>();
    self_correcting_axis_manager.run_passes(model);

    // ---- new genai inputs: input_ids / attention_mask / position_ids [b, seq] i64 ----
    auto input_ids = make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(input_ids, "input_ids");
    auto attention_mask = make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(attention_mask, "attention_mask");
    auto position_ids = make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(position_ids, "position_ids");

    // beam_idx (i32 [D]) is added by the make-stateful pass, next to the Gather that reads it; genai
    // sets it via set_tensor("beam_idx"). Keep that Parameter so its wiring is preserved. Its absence
    // means the model is not stateful, which the genai contract requires.
    auto beam_idx = find_parameter(model, "beam_idx");
    OPENVINO_ASSERT(beam_idx,
                    "[gguf] AdaptToGenAI: model has no 'beam_idx' input, so it is not stateful. "
                    "Register a make-stateful transformation extension (e.g. "
                    "ov::frontend::gguf::pass::MakeStateful) before converting.");

    // ---- token_len_per_seq = number of tokens in input_ids -> [1] ----
    // The token count is the ELEMENT COUNT of input_ids, not any single dimension of it. genai feeds
    // [batch, seq] (batch == 1), but SDPAToPagedAttention rewrites this Parameter to rank-1 [tokens]
    // and splices an Unsqueeze(axis=1) in front of its consumers, making it [tokens, 1]. Reading
    // dim 1 would then yield 1 for every prompt, collapsing the causal mask and the logits to a
    // single token; reading dim 0 breaks the un-rewritten case. ReduceProd is correct under both.
    auto ids_shape = make_shared<v3::ShapeOf>(input_ids, ov::element::i64);
    auto reduce_axis_0 = v0::Constant::create(ov::element::i64, {1}, {0});
    auto seq_len = make_shared<v1::ReduceProd>(ids_shape, reduce_axis_0, true);  // [1]
    token_len_per_seq->output(0).replace(seq_len->output(0));

    // The two gguf rank-4 input kinds carry the (batch, tokens) pair on different axes, so they get
    // different lifts. Both are written so the genai Parameter's own leading dims flow through
    // instead of being replaced by literals -- that is the whole mechanism by which one graph serves
    // both attention backends (see the layout note on the class).
    //
    // INDEX vectors (inp_tokens, inp_out_ids): consumed by get_rows, which squeezes the two leading
    // axes and gathers rows. The gathered result inherits the indices' trailing 2D shape, so the
    // indices must present exactly the Parameter's own [batch, seq]: prepend two 1s.
    //   SDPA: [1,1,1,tokens] -> squeeze -> [1,tokens] -> embd [1,tokens,n_embd]
    //   PA  : [1,1,tokens,1] -> squeeze -> [tokens,1] -> embd [tokens,1,n_embd]
    auto ones_1_1 = v0::Constant::create(ov::element::i64, {2}, {1, 1});
    const auto shape_1_1_batch_seq = make_shared<v0::Concat>(ov::OutputVector{ones_1_1, ids_shape}, 0);

    // ACTIVATION-like inputs (inp_pos): consumed by make_sin_cos, which transposes {0,3,1,2} to put
    // the token axis at 1, yielding cos/sin [batch, tokens, 1, n_rot/2] that broadcast against the
    // roped [batch, heads, tokens, head_size]. special_zero's 0 copies dim 0 and -1 absorbs the rest.
    //   SDPA: [1,1,1,tokens] -> cos/sin [1,tokens,1,half]
    //   PA  : [tokens,1,1,1] -> cos/sin [tokens,1,1,half], which broadcasts against [tokens,H,1,S]
    const auto shape_keep0_1_1_rest = v0::Constant::create(ov::element::i64, {4}, {0, 1, 1, -1});

    auto tokens_i32 = make_shared<v0::Convert>(input_ids, ov::element::i32);
    auto tokens_4d = make_shared<v1::Reshape>(tokens_i32, shape_1_1_batch_seq, false);
    inp_tokens->output(0).replace(tokens_4d->output(0));

    ov::Output<ov::Node> pos_i32 = make_shared<v0::Convert>(position_ids, ov::element::i32);
    // M-RoPE (qwen35): inp_pos carries FOUR position sections per token, laid out section-major --
    // make_sin_cos reshapes it to {..,4,tokens} and transposes. GenAI supplies one position per
    // token, so tile it 4x along the token axis. All four sections hold the same value here: the
    // per-section split only differs for image/video input, and a text-only prompt has no spatial
    // axes to differ on (llama.cpp fills all sections with the text position likewise).
    if (model->get_rt_info().count(gguf_imrope_key())) {
        auto tile_repeats = v0::Constant::create(ov::element::i64, {2}, {1, 4});
        pos_i32 = make_shared<v0::Tile>(pos_i32, tile_repeats);
    }
    auto pos_4d = make_shared<v1::Reshape>(pos_i32, shape_keep0_1_1_rest, true);
    inp_pos->output(0).replace(pos_4d->output(0));

    // ---- self_kq_mask [1,1,seq,kv_len] f32: 0 where attended, -inf above causal ----
    // kv_len = attention_mask length (= past + seq). query absolute positions = position_ids[0].
    auto am_shape = make_shared<v3::ShapeOf>(attention_mask, ov::element::i64);
    ov::Output<ov::Node> kv_len = get_dimensions(am_shape, {1});  // [1]

    // Flatten position_ids to [seq] via a shape-independent Reshape({-1}) rather than Squeeze(axis=0):
    // PA also rewrites position_ids to rank-1 and Unsqueezes it to [seq,1], where squeezing axis 0
    // would fail (or drop the wrong axis).
    auto flat_shape = v0::Constant::create(ov::element::i64, {1}, {-1});
    auto q_pos = make_shared<v0::Convert>(make_shared<v1::Reshape>(position_ids, flat_shape, false),
                                          ov::element::i32);  // [seq]
    auto one_1 = v0::Constant::create(ov::element::i64, {1}, {1});
    auto q_pos_col = make_shared<v1::Reshape>(q_pos,
                                              make_shared<v0::Concat>(ov::OutputVector{seq_len, one_1}, 0),
                                              false);  // [seq, 1]

    auto zero_i32 = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto one_i32 = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto squeeze_axis_0 = v0::Constant::create(ov::element::i64, {1}, {0});
    auto kv_len_i32 = make_shared<v0::Squeeze>(make_shared<v0::Convert>(kv_len, ov::element::i32),
                                               squeeze_axis_0);                              // scalar
    auto k_range = make_shared<v4::Range>(zero_i32, kv_len_i32, one_i32, ov::element::i32);  // [kv_len]
    auto k_row = make_shared<v1::Reshape>(k_range,
                                          make_shared<v0::Concat>(ov::OutputVector{one_1, kv_len}, 0),
                                          false);  // [1, kv_len]

    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto neg_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
    // [seq, kv_len] boolean predicate -> [1, 1, seq, kv_len] f32 mask (0 where attended, -inf elsewhere).
    auto to_mask_4d = [&](const ov::Output<ov::Node>& allowed_pred) {
        auto mask2d = make_shared<v1::Select>(allowed_pred, zero_f, neg_f);  // [seq, kv_len] f32
        return make_shared<v1::Reshape>(mask2d,
                                        make_shared<v0::Concat>(ov::OutputVector{ones_1_1, seq_len, kv_len}, 0),
                                        false);  // [1, 1, seq, kv_len]
    };

    auto allowed = make_shared<v1::LessEqual>(k_row, q_pos_col);  // [seq, kv_len] bool
    auto mask_4d = to_mask_4d(allowed);
    self_kq_mask->output(0).replace(mask_4d->output(0));

    // Sliding-window mask: for prompts within the window this equals the full causal mask, but
    // once the context (prompt + generated tokens) exceeds it, reusing the causal mask would
    // leave every older key visible and produce wrong logits. When the model's metadata records
    // an explicit window length (see gguf_swa_window_key), AND the causal mask with "key not
    // more than window - 1 steps behind the query"; a token at position q may attend to keys in
    // [q - window + 1, q]. Absent a recorded length (e.g. gpt-oss/gemma4, whose SWA is described
    // by sinks / a per-layer pattern with no accompanying token count here), fall back to the
    // full causal mask, matching the previous behavior.
    if (auto self_kq_mask_swa = find_parameter(model, "self_kq_mask_swa")) {
        ov::Output<ov::Node> swa_mask_4d = mask_4d->output(0);
        const auto& rt_info = model->get_rt_info();
        const auto swa_it = rt_info.find(gguf_swa_window_key());
        if (swa_it != rt_info.end()) {
            const auto window = swa_it->second.as<int64_t>();
            auto window_m1 = v0::Constant::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(window - 1)});
            auto window_start = make_shared<v1::Subtract>(q_pos_col, window_m1);      // [seq, 1]
            auto within_window = make_shared<v1::GreaterEqual>(k_row, window_start);  // [seq, kv_len]
            auto allowed_swa = make_shared<v1::LogicalAnd>(allowed, within_window);   // [seq, kv_len]
            swa_mask_4d = to_mask_4d(allowed_swa);
        }
        self_kq_mask_swa->output(0).replace(swa_mask_4d);
    }

    // inp_out_ids selects which rows the output head runs on. Emit the LAST row only: genai reads
    // just the final token's logits, so projecting every prompt position to vocab costs an extra
    // (tokens - 1) x hidden x vocab matmul per prefill.
    //
    // FixInpOutIdsRowSelect (above) already rewrote get_rows' original per-batch-row Gather into a
    // flat, global one over [batch*seq, hidden], so the index this parameter carries must be a
    // GLOBAL row index too: batch_index * seq_dim + (seq_dim - 1). Default ids_shape is [1, tokens]
    // (batch_dim=1), giving the single global index tokens - 1. Under SDPAToPagedAttention it is
    // [tokens, 1] (batch_dim=tokens, seq_dim=1), giving indices [0, 1, .., tokens - 1] -- the
    // identity that layout needs, since it already carries one token per row.
    if (auto inp_out_ids = find_parameter(model, "inp_out_ids")) {
        ov::Output<ov::Node> batch_dim = get_dimensions(ids_shape, {0});  // [1]: ids_shape[0]
        auto seq_dim = get_dimensions(ids_shape, {1});                    // [1]: ids_shape[1]
        auto seq_dim_i32 = make_shared<v0::Convert>(seq_dim, ov::element::i32);
        auto last_index =
            make_shared<v1::Subtract>(seq_dim_i32,
                                      v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}));  // [1]: seq_dim - 1
        auto batch_dim_scalar =
            make_shared<v0::Squeeze>(make_shared<v0::Convert>(batch_dim, ov::element::i32), squeeze_axis_0);
        auto row_ids = make_shared<v4::Range>(zero_i32,
                                              batch_dim_scalar,
                                              one_i32,
                                              ov::element::i32);  // [batch_dim]: 0 .. batch_dim - 1
        auto global_index = make_shared<v1::Add>(make_shared<v1::Multiply>(row_ids, seq_dim_i32),
                                                 last_index);  // [batch_dim]
        auto out_grid = make_shared<v1::Reshape>(global_index,
                                                 make_shared<v0::Concat>(ov::OutputVector{batch_dim, one_1}, 0),
                                                 false);  // [batch, 1]
        auto out_ids =
            make_shared<v1::Reshape>(out_grid,
                                     make_shared<v0::Concat>(ov::OutputVector{ones_1_1, batch_dim, one_1}, 0),
                                     false);
        inp_out_ids->output(0).replace(out_ids->output(0));
    }

    // ---- logits: rank-4 [.., .., .., vocab] -> [b, seq, vocab] ----
    // genai always wants [batch, seq, vocab] regardless of which axis the body kept the tokens on,
    // and both layouts hold seq*vocab contiguous values, so collapse everything ahead of vocab into
    // the sequence axis with a fixed batch of 1. (batch > 1 is not part of the genai stateful
    // contract this pass targets; token_len_per_seq above is likewise a whole-input token count.)
    // Keep ownership while add_results() may reallocate the model's ResultVector below.
    const std::shared_ptr<ov::op::v0::Result> old_result = model->get_results()[0];
    auto logits_src = old_result->input_value(0);
    auto vocab = get_dimensions(logits_src, {-1});  // [1]
    auto batch_seq_flat = v0::Constant::create(ov::element::i64, {2}, {1, -1});
    auto logits_3d = make_shared<v1::Reshape>(logits_src,
                                              make_shared<v0::Concat>(ov::OutputVector{batch_seq_flat, vocab}, 0),
                                              false);  // [1, seq, vocab]
    name_output(logits_3d, "logits");
    auto new_result = make_shared<v0::Result>(logits_3d);
    new_result->set_friendly_name("logits");

    model->add_results({new_result});
    model->remove_result(old_result);

    // Swap the input list to the genai contract. beam_idx is kept as-is; every other old
    // gguf Parameter has had its output rewired (consumers now read the derived subgraph),
    // so removing it is safe.
    model->add_parameters({input_ids, attention_mask, position_ids});
    const auto params_snapshot = model->get_parameters();  // copy: remove_parameter mutates the list
    for (const auto& p : params_snapshot) {
        if (p == input_ids || p == attention_mask || p == position_ids || p == beam_idx) {
            continue;
        }
        model->remove_parameter(p);
    }

    // Pin the runtime KV-cache precision to f16 for large-head models so decode matches both
    // prefill and llama.cpp; mainstream small-head models keep the faster u8 default. This is a
    // consumer-side optimization policy (genai), not a property the frontend bakes into the model.
    constexpr int64_t kU8SafeHeadSize = 128;
    if (max_kv_cache_head_size(model) > kU8SafeHeadSize) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
