// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_mask_preprocess_fusion.hpp"

#include <cstdint>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/causal_mask_preprocess.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::gen_pattern;

class CausalMaskPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CausalMaskPreprocess");
    CausalMaskPreprocess();

private:
    std::shared_ptr<ov::opset1::Constant> m_global_triu;
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
bool is_triu(ov::opset1::Constant* cmask, size_t rows, size_t columns) {
    const auto* ptr = reinterpret_cast<const T*>(cmask->get_data_ptr());
    for (size_t y = 0; y < rows; y++, ptr += columns) {
        size_t x;
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

    auto const_triu = makePattern<ov::opset1::Constant>({}, {});
    auto attention_mask = makePattern("i32[?,?]");
    auto batch_size = makePattern("i32[1]");
    auto cache_positions = makePattern("i32[?]");
    auto kvLen = makePattern("i32[1]");

    auto max_seq_len = Symbol("max_seq_len");

    const auto& ShapeOf_41610 = batch_size;  // shapeOf(beamidx)
    auto ListConstruct_Concat =
        makePattern<ov::opset1::Concat>({ShapeOf_41610, {1}, {1}, {1}}, {{"axis", 0}});  //  tensor_array<i32[4]>
    auto repeat_Tile =
        makePattern<ov::opset1::Tile>({const_triu, ListConstruct_Concat});  //  tensor_array<u8[?,1,8192,8192]>
    auto to_Convert =
        makePattern<ov::opset1::Convert>({repeat_Tile},
                                         {{"destination_type", "f32"}});  //  tensor_array<f32[?,1,8192,8192]>
    auto Constant_107277 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-FLT_MAX});
    auto mul_Multiply_1 =
        makePattern<ov::opset1::Multiply>({to_Convert, Constant_107277},
                                          {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,1,8192,8192]>
    auto SliceAssign_201_Reshape_0 =
        makePattern<ov::opset1::Reshape>({mul_Multiply_1, {-1}}, {{"special_zero", false}});  //  tensor_array<f32[?]>
    auto SliceAssign_201_ShapeOf = makePattern<ov::opset1::ShapeOf>({mul_Multiply_1});        //  tensor_array<i32[4]>
    auto SliceAssign_201_ReduceProd =
        makePattern<ov::opset1::ReduceProd>({SliceAssign_201_ShapeOf, 0},
                                            {{"keep_dims", false}});  //  tensor_array<i32[]>
    auto SliceAssign_201_Range = makePattern<ov::opset4::Range>({0, SliceAssign_201_ReduceProd, 1},
                                                                {{"output_type", "i32"}});  //  tensor_array<i32[?]>
    auto SliceAssign_201_Reshape =
        makePattern<ov::opset1::Reshape>({SliceAssign_201_Range, {-1, 1, max_seq_len, max_seq_len}},
                                         {{"special_zero", true}});  //  tensor_array<i32[?,1,8192,8192]>

    auto ShapeOf_49034 = makePattern<ov::opset1::ShapeOf>({attention_mask});  //  tensor_array<i32[2]>
    auto Gather_41642 =
        makePattern<ov::opset8::Gather>({ShapeOf_49034, {1}, 0}, {{"batch_dims", 0}});  //  tensor_array<i32[1]>
    auto ScatterUpdate_93502 =
        makePattern<ov::opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, Gather_41642, {0}});  //  tensor_array<i32[4]>
    auto SliceAssign_201_Slice = makePattern<ov::opset8::Slice>({SliceAssign_201_Reshape, {0}, Gather_41642, {1}, {3}});
    auto SliceAssign_201_StridedSlice = GenStridedSlice(SliceAssign_201_Reshape,
                                                        {0, 0, 0, 0},
                                                        ScatterUpdate_93502,
                                                        {1, 1, 1, 1},
                                                        3);  //  tensor_array<i32[?,1,8192,..8192]>
    auto SliceAssign_201_Reshape_1 =
        makePattern<ov::opset1::Reshape>({SliceAssign_201_Slice | SliceAssign_201_StridedSlice, {-1, 1}},
                                         {{"special_zero", false}});  //  tensor_array<i32[?,1]>
    auto causal_mask_boolean_slice = makePattern<ov::opset8::Slice>({mul_Multiply_1, {0}, Gather_41642, {1}, {3}});
    auto causal_mask_boolean_strided_slice = GenStridedSlice(mul_Multiply_1,
                                                             {0, 0, 0, 0},
                                                             ScatterUpdate_93502,
                                                             {1, 1, 1, 1},
                                                             3);  //  tensor_array<f32[?,1,8192,..8192]>
    auto Constant_107278 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {0.000000f});
    auto eq_Equal =
        makePattern<ov::opset1::Equal>({causal_mask_boolean_slice | causal_mask_boolean_strided_slice, Constant_107278},
                                       {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,8192,..8192]>
    auto unsqueeze_Unsqueeze_1 =
        makePattern<ov::opset1::Unsqueeze>({attention_mask, {1, 2}});  //  tensor_array<i32[?,1,1,?]>
    auto eq_Convert = makePattern<ov::opset1::Convert>({unsqueeze_Unsqueeze_1},
                                                       {{"destination_type", "f32"}});  //  tensor_array<f32[?,1,1,?]>
    auto Constant_107279 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {0.000000f});
    auto eq_Equal_1 = makePattern<ov::opset1::Equal>({eq_Convert, Constant_107279},
                                                     {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,1,?]>
    auto mul_LogicalAnd =
        makePattern<ov::opset1::LogicalAnd>({eq_Equal, eq_Equal_1},
                                            {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,8192,?]>
    auto masked_fill_Select = makePattern<ov::opset1::Select>(
        {mul_LogicalAnd, -FLT_MAX, causal_mask_boolean_slice | causal_mask_boolean_strided_slice},
        {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,1,8192,?]>
    auto copy_ShapeOf = makePattern<ov::opset1::ShapeOf>(
        {causal_mask_boolean_slice | causal_mask_boolean_strided_slice});  //  tensor_array<i32[4]>
    auto Constant_47319 = makeConst(ov::element::u8, ov::Shape({}), {0});
    auto copy_Broadcast =
        makePattern<ov::opset1::Broadcast>({masked_fill_Select, copy_ShapeOf, Constant_47319},
                                           {{"mode", "numpy"}});  //  tensor_array<f32[?,1,8192,..8192]>
    auto SliceAssign_201_Reshape_2 =
        makePattern<ov::opset1::Reshape>({copy_Broadcast, {-1}}, {{"special_zero", false}});  //  tensor_array<f32[?]>
    auto SliceAssign_201_ScatterNDUpdate = makePattern<ov::opset4::ScatterNDUpdate>(
        {SliceAssign_201_Reshape_0, SliceAssign_201_Reshape_1, SliceAssign_201_Reshape_2});  //  tensor_array<f32[?]>
    auto SliceAssign_201_Reshape_3 =
        makePattern<ov::opset1::Reshape>({SliceAssign_201_ScatterNDUpdate, {-1, 1, max_seq_len, max_seq_len}},
                                         {{"special_zero", true}});  //  tensor_array<f32[?,1,8192,8192]>
    auto ScatterUpdate_93554 =
        makePattern<ov::opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, kvLen, {0}});  //  tensor_array<i32[4]>
    auto slice_StridedSlice_14 = GenStridedSlice(SliceAssign_201_Reshape_3,
                                                 {0, 0, 0, 0},
                                                 ScatterUpdate_93554,
                                                 {1, 1, 1, 1},
                                                 3);  //  tensor_array<f32[?,1,8192,..8192]>
    auto slice_Slice_14 = makePattern<ov::opset8::Slice>({SliceAssign_201_Reshape_3, {0}, kvLen, {1}, {3}});
    auto index_Gather = makePattern<ov::opset8::Gather>({slice_Slice_14 | slice_StridedSlice_14, cache_positions, 2},
                                                        {{"batch_dims", 0}},
                                                        nullptr);  //  tensor_array<f32[?,1,?,..8192]>
    auto result = index_Gather;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }
        ov::intel_cpu::CausalMaskPreprocessNode::Config config;
        config.type = "CausalMaskPreprocess";

        auto triu = ov::as_type_ptr<ov::opset1::Constant>(pattern_map.find(const_triu)->second.get_node_shared_ptr());

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

        ov::OutputVector inputs{
            pattern_map.find(attention_mask)->second,
            pattern_map.find(batch_size)->second,
            pattern_map.find(cache_positions)->second,
            pattern_map.find(kvLen)->second,
        };
        auto replacement = std::make_shared<ov::intel_cpu::CausalMaskPreprocessNode>(inputs, config);
        ov::replace_node(root, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::CausalMaskPreprocessFusion::CausalMaskPreprocessFusion() {
    add_matcher<CausalMaskPreprocess>();
}
