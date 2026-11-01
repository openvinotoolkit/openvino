// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for the 4D float (additive) attention_mask contract of the NPUW LLM pipeline: how the mask is detected,
// which options reject it, how its shape is made static, how the infer request stages it for prefill and
// generate, and that an exported and imported model keeps the contract.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/data_utils.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

using ov::npuw::util::find_port_by_name;
using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::build_whisper_decoder_test_model;
using ov::test::npuw::LLMConfig;
using ov::test::npuw::make_test_model_config;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::RecordingFactory;
using ov::test::utils::make_tensor_with_precision_convert;

class TestableLLMCompiledModel final : public ov::npuw::LLMCompiledModel {
public:
    using ov::npuw::LLMCompiledModel::LLMCompiledModel;

    bool attention_mask_is_4d_float() const {
        return m_attention_mask_is_4d_float;
    }
};

// The test model config with a 4D float [B, 1, Q, K] attention_mask of the given type.
LLMConfig make_4d_float_mask_config(const ov::element::Type& type) {
    auto config = make_test_model_config();
    config.attention_mask_type = type;
    config.attention_mask_shape = ov::PartialShape{-1, 1, -1, -1};
    return config;
}

ov::AnyMap make_props(const ov::AnyMap& extra = {}) {
    ov::AnyMap props{{"NPUW_LLM", "YES"},
                     {"NPUW_DEVICES", "CPU"},
                     {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                     {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                     {"NPUW_LLM_PREFILL_HINT", "STATIC"}};
    for (const auto& [key, value] : extra) {
        props[key] = value;
    }
    return props;
}

class LLM4DFloatAttentionMaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    std::unique_ptr<TestableLLMCompiledModel> compile(const std::shared_ptr<ov::Model>& model,
                                                      const ov::AnyMap& props) {
        return std::make_unique<TestableLLMCompiledModel>(model, m_plugin, props, m_recorder.make_factory());
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    RecordingFactory m_recorder;
};

TEST_F(LLM4DFloatAttentionMaskTest, Binary2DMaskIsNotDetected) {
    std::unique_ptr<TestableLLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = compile(build_llm_test_model(), make_props()));
    EXPECT_FALSE(compiled->attention_mask_is_4d_float());
}

TEST_F(LLM4DFloatAttentionMaskTest, Rank2MaskIsNotDetectedWhateverItsElementType) {
    auto config = make_test_model_config();
    config.attention_mask_type = ov::element::f32;

    std::unique_ptr<TestableLLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = compile(build_llm_test_model(config), make_props()));
    EXPECT_FALSE(compiled->attention_mask_is_4d_float());
}

TEST_F(LLM4DFloatAttentionMaskTest, ModelWithoutAttentionMaskIsNotDetected) {
    std::unique_ptr<TestableLLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = compile(build_whisper_decoder_test_model(),
                                       {{"NPUW_LLM", "YES"},
                                        {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                        {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                        {"NPUW_WHISPER", "YES"},
                                        {"NPUW_WHISPER_EOS_TOKEN", "42"}}));
    EXPECT_FALSE(compiled->attention_mask_is_4d_float());
}

TEST_F(LLM4DFloatAttentionMaskTest, RejectsMaskContractsOtherThanRank2OrFloatRank4) {
    struct Case {
        const char* what;
        ov::element::Type type;
        ov::PartialShape shape;
    };
    const std::vector<Case> cases = {
        {"rank 3 float", ov::element::f32, ov::PartialShape{-1, -1, -1}},
        {"rank 4 boolean", ov::element::boolean, ov::PartialShape{-1, 1, -1, -1}},
        {"dynamic rank float", ov::element::f32, ov::PartialShape::dynamic()},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.what);
        auto config = make_4d_float_mask_config(test_case.type);
        config.attention_mask_shape = test_case.shape;
        EXPECT_THROW(compile(build_llm_test_model(config), make_props()), ov::Exception);
    }
}

// Each option below is accepted with the 2D int64 mask and rejected with NotImplemented for 4D float.
// Change when 4d float is supported for these features
TEST_F(LLM4DFloatAttentionMaskTest, ChunkedPrefillIsRejected) {
    const auto props = make_props({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}});
    EXPECT_NO_THROW(compile(build_llm_test_model(), props));
    EXPECT_THROW(compile(build_llm_test_model(make_4d_float_mask_config(ov::element::f32)), props), ov::NotImplemented);
}

TEST_F(LLM4DFloatAttentionMaskTest, PrefixCachingIsRejected) {
    const auto props = make_props({{"NPUW_LLM_ENABLE_PREFIX_CACHING", "YES"}});
    EXPECT_NO_THROW(compile(build_llm_test_model(), props));
    EXPECT_THROW(compile(build_llm_test_model(make_4d_float_mask_config(ov::element::f32)), props), ov::NotImplemented);
}

TEST_F(LLM4DFloatAttentionMaskTest, BlockBasedKVCacheIsRejected) {
    const auto props = make_props({{"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "YES"}});
    EXPECT_NO_THROW(compile(build_llm_test_model(), props));
    EXPECT_THROW(compile(build_llm_test_model(make_4d_float_mask_config(ov::element::f32)), props), ov::NotImplemented);
}

TEST_F(LLM4DFloatAttentionMaskTest, TextEmbeddingIsRejected) {
    const auto props = make_props({{"NPUW_TEXT_EMBED", "YES"}});
    auto config = make_test_model_config();
    config.use_kv_cache = false;
    config.internal_position_ids = true;
    config.lm_head_weight = {};
    EXPECT_NO_THROW(compile(build_llm_test_model(config), props));

    config.attention_mask_type = ov::element::f32;
    config.attention_mask_shape = ov::PartialShape{-1, 1, -1, -1};
    EXPECT_THROW(compile(build_llm_test_model(config), props), ov::NotImplemented);
}

class TestableLLMInferRequest final : public ov::npuw::LLMInferRequest {
public:
    explicit TestableLLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
        : ov::npuw::LLMInferRequest(compiled_model) {}

    ov::SoPtr<ov::ITensor> prefill_attention_mask() const {
        return m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
    }

    ov::SoPtr<ov::ITensor> generate_attention_mask() const {
        return m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask));
    }
};

// A distinct, negative value for every (row, column) of a mask up to kMarkerSize x kMarkerSize. The largest
// magnitude is 256, which f32, f16 and bf16 all represent exactly. The masks in these tests are small on purpose.
constexpr size_t kMarkerSize = 16;

float marker(size_t row, size_t column) {
    return -static_cast<float>(1 + row * kMarkerSize + column);
}

// An f32 tensor [1, 1, rows, columns] with value(row, column) in every position.
ov::Tensor make_tensor_values(size_t rows, size_t columns, const std::function<float(size_t, size_t)>& value) {
    ov::Tensor values(ov::element::f32, ov::Shape{1, 1, rows, columns});
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            values.data<float>()[row * columns + column] = value(row, column);
        }
    }
    return values;
}

ov::Tensor make_marker_mask(const ov::element::Type& type, size_t rows, size_t columns) {
    OPENVINO_ASSERT(rows <= kMarkerSize && columns <= kMarkerSize,
                    "marker() values are distinct and exact only for masks up to ",
                    kMarkerSize,
                    " x ",
                    kMarkerSize);
    return make_tensor_with_precision_convert(make_tensor_values(rows, columns, marker), type);
}

ov::Tensor make_i64(std::initializer_list<size_t> shape, int64_t fill_value) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), fill_value);
    return tensor;
}

ov::Tensor make_i64_iota(std::initializer_list<size_t> shape, int64_t start) {
    ov::Tensor tensor(ov::element::i64, ov::Shape(shape));
    std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), start);
    return tensor;
}

class LLM4DFloatAttentionMaskStagingTest : public LLM4DFloatAttentionMaskTest {
protected:
    void compile_with_mask_type(const ov::element::Type& type) {
        m_type = type;
        m_compiled = compile(build_llm_test_model(make_4d_float_mask_config(type)), make_props());
    }

    TestableLLMInferRequest& request() {
        if (!m_request) {
            m_request = std::make_unique<TestableLLMInferRequest>(m_compiled);
        }
        return *m_request;
    }

    void set_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) {
        const auto& inputs = request().get_inputs();
        request().set_tensor(find_port_by_name(inputs, "input_ids").value(), ov::get_tensor_impl(input_ids));
        request().set_tensor(find_port_by_name(inputs, "attention_mask").value(), ov::get_tensor_impl(attention_mask));
        request().set_tensor(find_port_by_name(inputs, "position_ids").value(), ov::get_tensor_impl(position_ids));
    }

    void run_prefill(size_t prompt_len) {
        set_inputs(make_i64({1, prompt_len}, 1),  // just a filler
                   make_marker_mask(m_type, prompt_len, prompt_len),
                   make_i64_iota({1, prompt_len}, 0));  // filled [0..prompt_len-1]
        request().infer();
    }

    // The caller mask covers the stored past and the current token: [1, 1, 1, past + 1].
    void run_generate_step(size_t past_len, size_t mask_keys) {
        set_inputs(make_i64({1, 1}, 1),  // just a filler
                   make_marker_mask(m_type, 1, mask_keys),
                   make_i64({1, 1}, static_cast<int64_t>(past_len)));
        request().infer();
    }

    ov::element::Type m_type;
    std::shared_ptr<TestableLLMCompiledModel> m_compiled;
    std::unique_ptr<TestableLLMInferRequest> m_request;
};

// The value the NPUW's LLMInferRequest uses for a fully masked position.
float lowest_of(const ov::element::Type& type) {
    switch (type) {
    case ov::element::f32:
        return std::numeric_limits<float>::lowest();
    case ov::element::f16:
        return static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    default:
        OPENVINO_THROW("Unsupported element type in the test: ", type);
    }
}

// Checks every cell of the staged mask and reports the first one that differs from expected(row, column).
void expect_cells(const ov::SoPtr<ov::ITensor>& staged, const std::function<float(size_t, size_t)>& expected) {
    const auto& shape = staged->get_shape();
    ASSERT_EQ(shape.size(), 4u);
    ASSERT_TRUE(staged->is_continuous());  // the conversion to f32 below reads the data linearly
    const auto actual = make_tensor_with_precision_convert(ov::make_tensor(staged), ov::element::f32);
    const auto wanted = make_tensor_values(shape[2], shape[3], expected);

    const float* wanted_begin = wanted.data<float>();
    const float* wanted_end = wanted_begin + wanted.get_size();
    const float* actual_begin = actual.data<float>();
    const auto [wanted_diff, actual_diff] = std::mismatch(wanted_begin, wanted_end, actual_begin);
    if (wanted_diff != wanted_end) {
        const size_t index = wanted_diff - wanted_begin;
        ADD_FAILURE() << "first mismatch at [" << index / shape[3] << ", " << index % shape[3] << "]: got "
                      << *actual_diff << ", expected " << *wanted_diff;
    }
}

class LLM4DFloatAttentionMaskByTypeTest : public LLM4DFloatAttentionMaskStagingTest,
                                          public ::testing::WithParamInterface<ov::element::Type> {
protected:
    void SetUp() override {
        LLM4DFloatAttentionMaskStagingTest::SetUp();
        compile_with_mask_type(GetParam());
    }
};

TEST_P(LLM4DFloatAttentionMaskByTypeTest, MaskIsDetectedAndMadeStatic) {
    EXPECT_TRUE(m_compiled->attention_mask_is_4d_float());

    const auto* prefill = m_recorder.find_contains("_prefill");
    const auto* generate = m_recorder.find_contains("_kv");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    // Prefill sees [1, 1, prompt, prompt]; generate sees [1, 1, 1, prompt + response].
    const auto prefill_mask = find_port_by_name(std::as_const(*prefill->model).inputs(), "attention_mask").value();
    const auto generate_mask = find_port_by_name(std::as_const(*generate->model).inputs(), "attention_mask").value();
    EXPECT_EQ(prefill_mask.get_partial_shape(), (ov::PartialShape{1, 1, 128, 128}));
    EXPECT_EQ(generate_mask.get_partial_shape(), (ov::PartialShape{1, 1, 1, 192}));
    EXPECT_EQ(prefill_mask.get_element_type(), GetParam());
    EXPECT_EQ(generate_mask.get_element_type(), GetParam());
}

// Prefill pads the caller mask on the left and on the top: its [Q, Q] block ends at the bottom-right corner of
// the static [P, P] mask, and every padded position is fully masked.
TEST_P(LLM4DFloatAttentionMaskByTypeTest, PrefillPutsTheMaskIntoTheBottomRightCorner) {
    constexpr size_t kPromptLen = 5;
    run_prefill(kPromptLen);

    const auto staged = request().prefill_attention_mask();
    const size_t size = staged->get_shape()[2];
    ASSERT_EQ(staged->get_shape(), (ov::Shape{1, 1, size, size}));
    const size_t offset = size - kPromptLen;
    const float masked = lowest_of(GetParam());
    expect_cells(staged, [&](size_t row, size_t column) {
        return (row >= offset && column >= offset) ? marker(row - offset, column - offset) : masked;
    });
}

// Generate puts the stored past on the left and the current token in the last column, whatever the static width.
TEST_P(LLM4DFloatAttentionMaskByTypeTest, GenerateSplitsTheMaskIntoPastAndCurrentColumns) {
    constexpr size_t kPromptLen = 5;
    run_prefill(kPromptLen);

    const float masked = lowest_of(GetParam());
    for (size_t step = 0; step < 2; ++step) {
        SCOPED_TRACE("generate step " + std::to_string(step));
        const size_t past_len = kPromptLen + step;
        run_generate_step(past_len, past_len + 1);

        const auto staged = request().generate_attention_mask();
        const size_t width = staged->get_shape()[3];
        ASSERT_EQ(staged->get_shape(), (ov::Shape{1, 1, 1, width}));
        expect_cells(staged, [&](size_t, size_t column) {
            if (column < past_len) {
                return marker(0, column);
            }
            return column == width - 1 ? marker(0, past_len) : masked;
        });
    }
}

TEST_P(LLM4DFloatAttentionMaskByTypeTest, GenerateRejectsMaskThatDisagreesWithTheStoredPast) {
    constexpr size_t kPromptLen = 5;
    run_prefill(kPromptLen);

    EXPECT_THROW(run_generate_step(kPromptLen, kPromptLen + 3), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(AdditiveMaskTypes,
                         LLM4DFloatAttentionMaskByTypeTest,
                         ::testing::Values(ov::element::f32, ov::element::f16),
                         [](const ::testing::TestParamInfo<ov::element::Type>& info) {
                             return info.param.to_string();
                         });

// The NPUW LLM pipeline converts bf16 to f16 for its sub-models, so the staged mask is f16 while the caller's is
// bf16, and the request rejects the mismatch on the first infer. Update this test when bf16 masks get supported.
TEST_F(LLM4DFloatAttentionMaskStagingTest, Bf16MaskIsRejectedAtInferBecauseSubmodelsUseF16) {
    compile_with_mask_type(ov::element::bf16);
    EXPECT_TRUE(m_compiled->attention_mask_is_4d_float());

    const auto* prefill = m_recorder.find_contains("_prefill");
    ASSERT_NE(prefill, nullptr);
    const auto prefill_mask = find_port_by_name(std::as_const(*prefill->model).inputs(), "attention_mask").value();
    EXPECT_EQ(prefill_mask.get_element_type(), ov::element::f16);

    EXPECT_THROW(run_prefill(5), ov::Exception);
}

// An additive causal mask [1, 1, rows, columns] for `rows` new tokens that follow `columns - rows` stored ones.
ov::Tensor make_causal_mask(size_t rows, size_t columns) {
    ov::Tensor mask(ov::element::f32, ov::Shape{1, 1, rows, columns});
    const size_t past_len = columns - rows;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            mask.data<float>()[row * columns + column] = column <= row + past_len ? 0.0f : lowest_of(ov::element::f32);
        }
    }
    return mask;
}

// The blob does not store whether the mask is 4D float: the import derives it from the restored inputs. The imported
// model must still take the caller's f32 mask for prefill and for generate and give the compiled model's logits. The
// model goes through disk like a converted model, so the blob is weightless and the import gets only the weights path.
// The prompt fills the static prompt window. With a shorter one, the CPU sub-models compute in bf16 by default and the
// f32 lowest value that NPUW stages for the padding rows turns into -inf there, so the logits are NaN. Requesting f32
// inference precision removes the NaN.
TEST(LLM4DFloatAttentionMaskRoundTrip, ImportedModelRunsLikeTheCompiledOne) {
    constexpr size_t kGenerateSteps = 2;

    ov::Core core;
    const auto dir = std::filesystem::temp_directory_path() / "npuw_4d_float_mask_round_trip";
    std::filesystem::create_directories(dir);
    const auto xml = dir / "model.xml";
    const auto bin = dir / "model.bin";
    ov::serialize(build_llm_test_model(make_4d_float_mask_config(ov::element::f32)), xml.string(), bin.string());
    const auto model = core.read_model(xml.string(), bin.string());

    // The online partitioner's repeated-block detection does not cope with the synthetic model, so the sub-models
    // are compiled whole on CPU.
    const auto props = make_props({{"NPU_USE_NPUW", "YES"}, {"NPUW_ONLINE_PIPELINE", "NONE"}});
    const auto prompt_len = props.at("NPUW_LLM_MAX_PROMPT_LEN").as<size_t>();
    auto compiled = core.compile_model(model, "NPU", props);

    std::stringstream blob;
    compiled.export_model(blob);
    auto import_props = props;
    import_props[ov::weights_path.name()] = bin.string();
    auto imported = core.import_model(blob, "NPU", import_props);

    // The prompt in one prefill, then single-token generate steps; returns the logits of every step.
    const auto run = [&](ov::CompiledModel& model_under_test) {
        auto request = model_under_test.create_infer_request();
        ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
        beam_idx.data<int32_t>()[0] = 0;
        std::vector<ov::Tensor> logits;
        const auto step = [&](size_t past_len, size_t new_tokens) {
            request.set_tensor("input_ids", make_i64_iota({1, new_tokens}, static_cast<int64_t>(1 + past_len)));
            request.set_tensor("attention_mask", make_causal_mask(new_tokens, past_len + new_tokens));
            request.set_tensor("position_ids", make_i64_iota({1, new_tokens}, static_cast<int64_t>(past_len)));
            request.set_tensor("beam_idx", beam_idx);
            request.infer();
            auto output = request.get_output_tensor(0);
            ov::Tensor copy(output.get_element_type(), output.get_shape());
            output.copy_to(copy);
            logits.push_back(make_tensor_with_precision_convert(copy, ov::element::f32));
        };
        step(0, prompt_len);
        for (size_t i = 0; i < kGenerateSteps; ++i) {
            step(prompt_len + i, 1);
        }
        return logits;
    };

    const auto expected = run(compiled);
    const auto actual = run(imported);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t step = 0; step < expected.size(); ++step) {
        SCOPED_TRACE("step " + std::to_string(step));
        ASSERT_EQ(actual[step].get_shape(), expected[step].get_shape());
        for (size_t i = 0; i < expected[step].get_size(); ++i) {
            ASSERT_NEAR(actual[step].data<float>()[i], expected[step].data<float>()[i], 1e-5f) << "element " << i;
        }
    }
}

}  // namespace
