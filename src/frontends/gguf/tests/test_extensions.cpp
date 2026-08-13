// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for FrontEnd::add_extension (the extension-passing path in frontend.cpp).
//
// Two extension kinds are covered:
//
// - ov::frontend::ConversionExtension registers a custom translator for a ggml op name; the
//   frontend merges it into the op table (overriding a built-in translator on name collision, or
//   adding a translator for an otherwise unsupported op). The converter receives an
//   ov::frontend::NodeContext, which the gguf NodeContext derives from.
//
// - ov::frontend::DecoderTransformationExtension registers a normalization pass, run ahead of the
//   frontend's built-in lowerings. This is how the EXECUTION MODE is chosen: conversion always
//   yields a stateless graph (KV caches as Parameter/Result pairs written by a SetRows
//   placeholder), and a caller that wants an OpenVINO KV cache registers
//   ov::frontend::gguf::pass::MakeStateful here, which consumes those SetRows ops before the
//   default stateless lowering ever sees them.

#include <algorithm>
#include <set>
#include <stdexcept>

#include "op_test_utils.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scatter_update.hpp"

using namespace ov_gguf_test;

namespace {

// A ConversionExtension whose converter emits Negative(in0) for whatever op it is
// registered against.
std::shared_ptr<ov::frontend::ConversionExtension> make_negate_ext(const std::string& op_type) {
    return std::make_shared<ov::frontend::ConversionExtension>(
        op_type,
        [](const ov::frontend::NodeContext& context) -> ov::OutputVector {
            return {std::make_shared<ov::op::v0::Negative>(context.get_input(0))};
        });
}

}  // namespace

// A ConversionExtension registered for a built-in op name overrides the built-in
// translator: GGML_OP_SCALE normally does in*scale+bias, but here it must negate.
TEST(GGUFExtensions, ConversionExtensionOverridesBuiltin) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<float>("scale", 2.0f)
                     .attr<float>("bias", 0.0f)
                     .build_with_extensions({make_negate_ext("GGML_OP_SCALE")});

    std::vector<float> x{1, -2, 3, -4, 5, -6, 7, -8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = -x[i];  // Negative, not the built-in scale
    expect_near(out, expected);
}

// A ConversionExtension can add a translator for an op the frontend does not support
// out of the box (here a made-up "GGML_OP_CUSTOM_NEGATE").
TEST(GGUFExtensions, ConversionExtensionAddsNewOp) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CUSTOM_NEGATE")
                     .input("x", ov::element::f32, {3})
                     .output("out", ov::element::f32, {3})
                     .build_with_extensions({make_negate_ext("GGML_OP_CUSTOM_NEGATE")});

    std::vector<float> x{1, -2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({3}, x)}});
    expect_near(out, {-1, 2, -3});
}

// Two extensions registered together are both applied.
TEST(GGUFExtensions, MultipleConversionExtensions) {
    auto abs_ext = std::make_shared<ov::frontend::ConversionExtension>(
        "GGML_OP_CUSTOM_ABS",
        [](const ov::frontend::NodeContext& context) -> ov::OutputVector {
            return {std::make_shared<ov::op::v0::Abs>(context.get_input(0))};
        });

    auto model = SingleOpBuilder()
                     .op("GGML_OP_CUSTOM_ABS")
                     .input("x", ov::element::f32, {4})
                     .output("out", ov::element::f32, {4})
                     .build_with_extensions({make_negate_ext("GGML_OP_CUSTOM_NEGATE"), abs_ext});

    std::vector<float> x{1, -2, 3, -4};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({4}, x)}});
    expect_near(out, {1, 2, 3, 4});  // Abs applied
}

// Without the extension, an unsupported op fails to convert -- confirming the op is not
// already known and that the extension in the test above is what enables it.
TEST(GGUFExtensions, UnsupportedOpWithoutExtensionThrows) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_CUSTOM_NEGATE")
                       .input("x", ov::element::f32, {3})
                       .output("out", ov::element::f32, {3});
    EXPECT_ANY_THROW(builder.build());
}

// ── DecoderTransformationExtension: choosing the execution mode ─────────────────────────────────

namespace {

// One GGML_OP_SET_ROWS writing `data` rows at `idx` into the `cache` input -- the shape of a KV
// cache write, in the layout the native .gguf builder emits: [1, tokens, n_head_kv, head_size],
// whose one dynamic axis (1, the token axis) is what MakeStateful infers the append axis from.
SingleOpBuilder kv_cache_write_builder() {
    return SingleOpBuilder()
        .op("GGML_OP_SET_ROWS")
        .input("data", ov::element::f32, {1, -1, 2, 4})
        .input("idx", ov::element::i64, {1, 1, 1, -1})
        .input("cache", ov::element::f16, {1, -1, 2, 4})
        .output("cache_out", ov::element::f16, {1, -1, 2, 4});
}

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_info() == type) {
            n++;
        }
    }
    return n;
}

}  // namespace

// The default: with no extension registered, conversion lowers every SetRows to the stateless
// ScatterUpdate form, and the cache stays an ordinary model input/output. This is the baseline the
// design rests on -- the frontend itself is stateless, like an optimum-intel export.
TEST(GGUFExtensions, NoExtensionYieldsStatelessCache) {
    auto model = kv_cache_write_builder().build();

    EXPECT_TRUE(model->get_variables().empty());
    EXPECT_TRUE(model->get_sinks().empty());
    EXPECT_EQ(count_ops_of_type(model, ov::op::v3::ScatterUpdate::get_type_info_static()), 1);
    // The SetRows placeholder is an internal op and must never survive conversion.
    EXPECT_EQ(count_ops_of_type(model, SetRows::get_type_info_static()), 0);
    // cache is still an input, cache_out still an output.
    EXPECT_EQ(model->get_parameters().size(), 3);
    EXPECT_EQ(model->get_results().size(), 1);

    // No beam_idx: it is a stateful-cache concept, so the stateless graph must not carry one. This is
    // what lets the native builder and a llama.cpp cgraph decoder agree on their stateless IO -- a
    // decoder-declared beam_idx would be an input with no consumer here.
    for (const auto& p : model->get_parameters()) {
        EXPECT_NE(p->get_friendly_name(), "beam_idx");
    }
}

// Registering MakeStateful as a DecoderTransformationExtension swaps the execution mode: the same
// conversion now yields an OpenVINO state. The cache Parameter/Result pair is gone, replaced by a
// Variable with a ReadValue/Concat/Assign, and no ScatterUpdate is emitted -- the extension ran
// ahead of the built-in stateless lowering and consumed the SetRows first.
TEST(GGUFExtensions, MakeStatefulExtensionYieldsStatefulCache) {
    auto model = kv_cache_write_builder().build_with_extensions(
        {std::make_shared<ov::frontend::DecoderTransformationExtension>(pass::MakeStateful())});

    ASSERT_EQ(model->get_variables().size(), 1);
    EXPECT_EQ(model->get_sinks().size(), 1);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v6::ReadValue::get_type_info_static()), 1);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v6::Assign::get_type_info_static()), 1);
    EXPECT_EQ(count_ops_of_type(model, ov::op::v3::ScatterUpdate::get_type_info_static()), 0);
    EXPECT_EQ(count_ops_of_type(model, SetRows::get_type_info_static()), 0);

    // The cache left the model's IO entirely: data + idx remain, and beam_idx was ADDED by the pass
    // (see below). The cache Result became the Assign sink.
    EXPECT_EQ(model->get_parameters().size(), 3);
    EXPECT_EQ(model->get_results().size(), 0);

    // beam_idx belongs to the state, so the pass creates it -- no decoder declares it. Its Gather on
    // the past is what CPU's stateful_sdpa_fusion matches.
    auto beam_idx = std::find_if(model->get_parameters().begin(),
                                 model->get_parameters().end(),
                                 [](const std::shared_ptr<ov::op::v0::Parameter>& p) {
                                     return p->get_friendly_name() == "beam_idx";
                                 });
    ASSERT_NE(beam_idx, model->get_parameters().end());
    EXPECT_EQ((*beam_idx)->get_element_type(), ov::element::i32);
    EXPECT_EQ((*beam_idx)->get_partial_shape(), ov::PartialShape({-1}));
    EXPECT_EQ(count_ops_of_type(model, ov::op::v8::Gather::get_type_info_static()), 1);

    // The Variable is named after the cache input and its append axis is dynamic (the state grows
    // by this step's rows on every inference), the rest keeping the cache's declared dims.
    const auto& info = model->get_variables()[0]->get_info();
    EXPECT_EQ(info.variable_id, "cache");
    EXPECT_EQ(info.data_type, ov::element::f16);
    EXPECT_EQ(info.data_shape, ov::PartialShape({1, -1, 2, 4}));
}

// skip_caches leaves a named cache stateless while other caches are converted. A sliding-window
// cache needs this: it is evicted from the front, not only appended to, so an append-grown Variable
// would not reproduce it.
TEST(GGUFExtensions, MakeStatefulSkipsNamedCache) {
    auto model = kv_cache_write_builder().build_with_extensions(
        {std::make_shared<ov::frontend::DecoderTransformationExtension>(pass::MakeStateful({"cache"}))});

    // The only cache was skipped, so the pass made no change and the built-in stateless lowering
    // handled the SetRows -- an identical result to registering no extension at all.
    EXPECT_TRUE(model->get_variables().empty());
    EXPECT_EQ(count_ops_of_type(model, ov::op::v3::ScatterUpdate::get_type_info_static()), 1);
    EXPECT_EQ(model->get_parameters().size(), 3);
    EXPECT_EQ(model->get_results().size(), 1);
}

// ── the stateless IO contract: two decoders of one model must agree ──────────────────────────────

namespace {

// A decoder that routes a named subset of its inputs through get_model_extra_inputs() instead of
// get_model_inputs(), which is the one structural difference between how the native .gguf builder
// and the llama.cpp cgraph decoder present a model's IO. Both halves land in the same graph, so
// converting either way must yield the same stateless inputs.
class SplitIoDecoder : public SingleOpDecoder {
public:
    SplitIoDecoder(const SingleOpDecoder& base, const std::set<std::string>& as_extra) : SingleOpDecoder(base) {
        for (const auto& name : as_extra) {
            auto it = m_split_main.find(name);
            if (it == m_split_main.end()) {
                throw std::runtime_error("SplitIoDecoder: no such input '" + name + "'");
            }
            m_split_extra[name] = it->second;
            m_split_main.erase(it);
        }
    }

    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_split_main;
    }
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override {
        return m_split_extra;
    }

private:
    // Seeded from the base decoder's inputs (member initializers run before the constructor body),
    // then partitioned by that body.
    std::map<std::string, std::shared_ptr<ov::Node>> m_split_main = SingleOpDecoder::get_model_inputs();
    std::map<std::string, std::shared_ptr<ov::Node>> m_split_extra;
};

std::set<std::string> input_names(const std::shared_ptr<ov::Model>& model) {
    std::set<std::string> names;
    for (const auto& p : model->get_parameters()) {
        names.insert(p->get_friendly_name());
    }
    return names;
}

}  // namespace

// The frontend invents no inputs of its own: the stateless graph's inputs are exactly what the
// decoder declared, however the decoder chose to split them between get_model_inputs() and
// get_model_extra_inputs(). That split is the one structural difference between the native builder
// and the llama.cpp cgraph decoder, so pinning it down here is half of "the two decoders produce the
// same graph"; the other half -- that neither decoder declares an input the other cannot, beam_idx
// being the case that got this wrong -- needs a real .gguf and lives in the model-level checks.
TEST(GGUFExtensions, StatelessIoIsExactlyTheDecoderInputs) {
    const std::set<std::string> declared{"data", "idx", "cache"};

    auto base = kv_cache_write_builder();
    auto via_main = base.build();
    EXPECT_EQ(input_names(via_main), declared);

    // The same op, with "cache" and "idx" presented as auxiliary inputs the way the cgraph decoder
    // presents its extras. Same graph inputs -> the two decoders agree.
    FrontEnd fe;
    auto split = std::make_shared<SplitIoDecoder>(*std::dynamic_pointer_cast<SingleOpDecoder>(base.decoder()),
                                                  std::set<std::string>{"cache", "idx"});
    auto via_extra = fe.convert(fe.load(std::static_pointer_cast<GgufDecoder>(split)));
    EXPECT_EQ(input_names(via_extra), declared);
}

// And making the model stateful adds exactly one input, beam_idx, on top of that contract -- so the
// stateful IO is a function of the pass, not of which decoder produced the stateless graph.
TEST(GGUFExtensions, MakeStatefulAddsOnlyBeamIdx) {
    auto stateless = input_names(kv_cache_write_builder().build());
    auto stateful = input_names(kv_cache_write_builder().build_with_extensions(
        {std::make_shared<ov::frontend::DecoderTransformationExtension>(pass::MakeStateful())}));

    // The cache Parameter became a Variable, and beam_idx appeared.
    stateless.erase("cache");
    stateless.insert("beam_idx");
    EXPECT_EQ(stateful, stateless);
}

// A DecoderTransformationExtension can hold any pass, not only the ones the frontend ships: here a
// plain lambda pass, which must run during conversion (it renames the model, observable after).
TEST(GGUFExtensions, ArbitraryTransformationExtensionRuns) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {2, 2})
                     .output("out", ov::element::f32, {2, 2})
                     .attr<float>("scale", 2.0f)
                     .attr<float>("bias", 0.0f)
                     .build_with_extensions({std::make_shared<ov::frontend::DecoderTransformationExtension>(
                         [](const std::shared_ptr<ov::Model>& m) {
                             m->set_friendly_name("touched_by_extension");
                             return true;
                         })});

    EXPECT_EQ(model->get_friendly_name(), "touched_by_extension");
}
