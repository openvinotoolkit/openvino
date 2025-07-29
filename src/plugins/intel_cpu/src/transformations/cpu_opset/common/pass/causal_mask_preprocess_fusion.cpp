// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_mask_preprocess_fusion.hpp"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/op.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;
using namespace ov::op;
using namespace ov;

class CausalMaskPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CausalMaskPreprocess");
    CausalMaskPreprocess();

private:
    std::shared_ptr<ov::op::v0::Constant> m_global_triu;
};

/*
following pattern is from:
    LlamaModel._update_causal_mask() models/llama/modeling_llama.py:
    GemmaModel._update_causal_mask() models/llama/modeling_gemma.py:

the python code is:

    min_dtype = torch.finfo(dtype).min
    causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype
    causal_mask = causal_mask.to(dtype=dtype, device=device)

    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

    ######## when being used will be further sliced
    causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

inputs:
    causal mask (slice_Slice) : boolean[1, 1, maxL, maxL]
    attention_mask            : i64[N, kv_len]
                                 0 means mask-out, 1 means attends to
    batch_size (size_Gather)  : i32[1]
    cache_positions  i32[q_len];
    kvLen            i32[1];

outputs:
    causal mask for SDPA : f32[batch_size, 1, q_len, kvLen]
*/

template <typename T>
bool is_triu(ov::op::v0::Constant* cmask, size_t rows, size_t columns) {
    const auto* ptr = reinterpret_cast<const T*>(cmask->get_data_ptr());
    for (size_t y = 0; y < rows; y++, ptr += columns) {
        size_t x = 0;
        for (x = 0; x <= y; x++) {
            if (ptr[x]) {
                return false;
            }
        }
        for (; x < columns; x++) {
            if (!ptr[x]) {
                return false;
            }
        }
    }
    return true;
}

CausalMaskPreprocess::CausalMaskPreprocess() {
    MATCHER_SCOPE(CausalMaskPreprocess);

    auto const_triu = pattern::wrap_const();
    auto attention_mask = pattern::any_input(pattern::type_matches(element::i32) && pattern::rank_equals(2));
    auto batch_size = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));
    auto cache_positions = pattern::any_input(pattern::type_matches(element::i32) && pattern::rank_equals(1));
    auto kvLen = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));

    auto concat0 = pattern::wrap_type<v0::Concat>({batch_size, {1}, {1}, {1}}, {{"axis", 0}});
    auto tile0 = pattern::wrap_type<v0::Tile>({const_triu, concat0});
    auto convert0 = pattern::wrap_type<v0::Convert>({tile0}, {{"destination_type", "f32"}});
    auto constant0 = pattern::wrap_type<v0::Constant>(pattern::type_matches(element::f32) &&
                                                      pattern::shape_matches("[1, 1, 1, 1]") &&
                                                      pattern::value_matches(std::to_string(-FLT_MAX)));
    auto multiply0 = pattern::wrap_type<v1::Multiply>({convert0, constant0}, {{"auto_broadcast", "numpy"}});
    auto reshape0 = pattern::wrap_type<v1::Reshape>({multiply0, {-1}}, {{"special_zero", false}});
    auto shape_of0 = pattern::wrap_type<v0::ShapeOf>({multiply0});
    auto reduce_prod0 = pattern::wrap_type<v1::ReduceProd>({shape_of0, 0}, {{"keep_dims", false}});

    auto range0 = pattern::wrap_type<v4::Range>({0, reduce_prod0, 1}, {{"output_type", "i32"}});
    auto reshape1 =
        pattern::wrap_type<v1::Reshape>({range0, {"-1", "1", "max_seq_len", "max_seq_len"}}, {{"special_zero", true}});

    auto shape_of1 = pattern::wrap_type<v0::ShapeOf>({attention_mask});
    auto gather0 = pattern::wrap_type<v8::Gather>({shape_of1, {1}, 0}, {{"batch_dims", 0}});
    auto scatter_update0 = pattern::wrap_type<v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, gather0, {0}});
    auto slice0 = pattern::wrap_type<v8::Slice>({reshape1, {0}, gather0, {1}, {3}});
    auto strided_slice0 = ov::op::util::NewGenStridedSlice(reshape1, {0, 0, 0, 0}, scatter_update0, {1, 1, 1, 1}, 3);
    auto reshape2 = pattern::wrap_type<v1::Reshape>({slice0 | strided_slice0, {-1, 1}}, {{"special_zero", false}});
    auto causal_mask_boolean_slice = pattern::wrap_type<v8::Slice>({multiply0, {0}, gather0, {1}, {3}});
    auto causal_mask_boolean_strided_slice =
        ov::op::util::NewGenStridedSlice(multiply0, {0, 0, 0, 0}, scatter_update0, {1, 1, 1, 1}, 3);
    auto constant1 = pattern::wrap_type<v0::Constant>(pattern::type_matches(ov::element::f32) &&
                                                      pattern::shape_matches("[1, 1, 1, 1]") &&
                                                      pattern::value_matches(std::to_string(0.0F)));
    auto equal0 =
        pattern::wrap_type<v1::Equal>({causal_mask_boolean_slice | causal_mask_boolean_strided_slice, constant1},
                                      {{"auto_broadcast", "numpy"}});
    auto unsqueeze0 = pattern::wrap_type<v0::Unsqueeze>({attention_mask, {1, 2}});
    auto convert1 = pattern::wrap_type<v0::Convert>({unsqueeze0}, {{"destination_type", "f32"}});
    auto constant2 = pattern::wrap_type<v0::Constant>(pattern::type_matches(ov::element::f32) &&
                                                      pattern::shape_matches("[1, 1, 1, 1]") &&
                                                      pattern::value_matches(std::to_string(0.0F)));
    auto equal1 = pattern::wrap_type<v1::Equal>({convert1, constant2}, {{"auto_broadcast", "numpy"}});
    auto and0 = pattern::wrap_type<v1::LogicalAnd>({equal0, equal1}, {{"auto_broadcast", "numpy"}});
    auto masked_fill_Select =
        pattern::wrap_type<v1::Select>({and0, -FLT_MAX, causal_mask_boolean_slice | causal_mask_boolean_strided_slice},
                                       {{"auto_broadcast", "numpy"}});
    auto shape_of2 = pattern::wrap_type<v0::ShapeOf>({causal_mask_boolean_slice | causal_mask_boolean_strided_slice});
    auto constant3 = pattern::wrap_type<v0::Constant>(pattern::type_matches(element::u8) &&
                                                      pattern::shape_matches("[]") && pattern::value_matches("0"));
    auto broadcast0 =
        pattern::wrap_type<v1::Broadcast>({masked_fill_Select, shape_of2, constant3}, {{"mode", "numpy"}});
    auto reshape3 = pattern::wrap_type<v1::Reshape>({broadcast0, {-1}}, {{"special_zero", false}});
    auto scatternd_update0 = pattern::wrap_type<v3::ScatterNDUpdate>({reshape0, reshape2, reshape3});
    auto reshape4 = pattern::wrap_type<v1::Reshape>({scatternd_update0, {"-1", "1", "max_seq_len", "max_seq_len"}},
                                                    {{"special_zero", true}});
    auto scatternd_update1 = pattern::wrap_type<v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, kvLen, {0}});
    auto strided_slice1 = ov::op::util::NewGenStridedSlice(reshape4, {0, 0, 0, 0}, scatternd_update1, {1, 1, 1, 1}, 3);
    auto slice1 = pattern::wrap_type<v8::Slice>({reshape4, {0}, kvLen, {1}, {3}});
    auto index_Gather =
        pattern::wrap_type<v8::Gather>({slice1 | strided_slice1, cache_positions, 2}, {{"batch_dims", 0}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        ov::intel_cpu::CausalMaskPreprocessNode::Config config;
        config.type = "CausalMaskPreprocess";

        auto const_triu_it = pattern_map.find(const_triu);
        if (const_triu_it == pattern_map.end()) {
            return false;
        }
        auto triu = ov::as_type_ptr<ov::op::v0::Constant>(const_triu_it->second.get_node_shared_ptr());
        if (!triu) {
            return false;
        }

        auto triu_shape = triu->get_output_shape(0);
        if (triu_shape.size() != 4) {
            return false;
        }
        if (triu_shape[0] != 1 || triu_shape[1] != 1 || triu_shape[2] != triu_shape[3]) {
            return false;
        }

        if (!m_global_triu) {
            auto triu_dtype = triu->get_output_element_type(0);
            // check if it's triu
            if (triu_dtype == ov::element::i32) {
                if (!is_triu<int32_t>(triu.get(), triu_shape[2], triu_shape[3])) {
                    return false;
                }
            } else if (triu_dtype == ov::element::u8) {
                if (!is_triu<uint8_t>(triu.get(), triu_shape[2], triu_shape[3])) {
                    return false;
                }
            } else {
                return false;
            }
            // there should be only 1 raw triu constant
            m_global_triu = triu;
        } else {
            // check identity insread of values to save time
            if (triu != m_global_triu) {
                return false;
            }
        }

        auto attention_mask_it = pattern_map.find(attention_mask);
        auto batch_size_it = pattern_map.find(batch_size);
        auto cache_positions_it = pattern_map.find(cache_positions);
        auto kvLen_it = pattern_map.find(kvLen);

        if (attention_mask_it == pattern_map.end() || batch_size_it == pattern_map.end() ||
            cache_positions_it == pattern_map.end() || kvLen_it == pattern_map.end()) {
            return false;
        }

        ov::OutputVector inputs{
            attention_mask_it->second,
            batch_size_it->second,
            cache_positions_it->second,
            kvLen_it->second,
        };
        auto replacement = std::make_shared<ov::intel_cpu::CausalMaskPreprocessNode>(inputs, config);
        ov::replace_node(root, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(index_Gather, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::intel_cpu::CausalMaskPreprocessFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(CausalMaskPreprocessFusion);

    SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();

    symbolic_ctx_manager->register_pass<CausalMaskPreprocess>();

    return symbolic_optimizations.run_on_model(model);
}
