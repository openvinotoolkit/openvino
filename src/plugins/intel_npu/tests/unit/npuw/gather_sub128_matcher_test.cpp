// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "lazy_tensor.hpp"
#include "intel_npu/config/npuw.hpp"
#include "npuw_transformations/insert_vocab_sub128.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"
#include "partitioning/partitioning.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {

std::shared_ptr<ov::Model> make_gather_model(float shift_value) {
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1});
    auto weights = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 2}, std::vector<uint8_t>(8, 200));
    auto zero_point = ov::opset10::Constant::create(ov::element::u8, ov::Shape{4, 1}, std::vector<uint8_t>(4, 128));
    auto scale = ov::opset10::Constant::create(ov::element::f16, ov::Shape{4, 1}, std::vector<float>(4, 1.0f));
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, ov::element::f16);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, ov::element::f16);
    auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {shift_value});
    auto shifted_weight = std::make_shared<ov::opset10::Subtract>(weight_convert, shift);
    auto shifted_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);
    if (shift_value == 128.0f) {
        shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        shifted_zero_point->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    }

    auto dequantized = std::make_shared<ov::opset10::Subtract>(shifted_weight, shifted_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    auto gathered = std::make_shared<ov::opset10::Gather>(converted, ids, axis);
    auto result = std::make_shared<ov::opset10::Result>(gathered);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});
}

std::size_t count_gathers(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Gather>(node)) {
            ++count;
        }
    }
    return count;
}

bool run_lift(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::DQLiftGatherAsymCW>();
    return rewrite.run_on_model(model);
}

std::shared_ptr<ov::Model> make_parameter_gather_model(std::optional<float> weight_shift,
                                                       std::optional<float> zero_point_shift) {
    constexpr std::size_t vocab_size = 4096;
    constexpr std::size_t hidden_size = 2048;
    auto ids = std::make_shared<ov::opset10::Parameter>(ov::element::i32, ov::Shape{1, 1});
    auto weights = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{vocab_size, hidden_size});
    auto zero_point = std::make_shared<ov::opset10::Parameter>(ov::element::u8, ov::Shape{vocab_size, hidden_size});
    auto scale = std::make_shared<ov::opset10::Parameter>(ov::element::f16, ov::Shape{vocab_size, hidden_size});
    auto axis = ov::opset10::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto gathered_weights = std::make_shared<ov::opset10::Gather>(weights, ids, axis);
    auto gathered_zero_point = std::make_shared<ov::opset10::Gather>(zero_point, ids, axis);
    auto gathered_scale = std::make_shared<ov::opset10::Gather>(scale, ids, axis);
    auto weight_convert = std::make_shared<ov::opset10::Convert>(gathered_weights, ov::element::f16);
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(gathered_zero_point, ov::element::f16);
    ov::Output<ov::Node> dequantized_weight = weight_convert;
    ov::Output<ov::Node> dequantized_zero_point = zero_point_convert;
    if (weight_shift.has_value()) {
        auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {weight_shift.value()});
        dequantized_weight = std::make_shared<ov::opset10::Subtract>(weight_convert, shift);
        if (weight_shift.value() == 128.0f) {
            dequantized_weight.get_node_shared_ptr()->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        }
    }
    if (zero_point_shift.has_value()) {
        auto shift = ov::opset10::Constant::create(ov::element::f16, ov::Shape{}, {zero_point_shift.value()});
        dequantized_zero_point = std::make_shared<ov::opset10::Subtract>(zero_point_convert, shift);
        if (zero_point_shift.value() == 128.0f) {
            dequantized_zero_point.get_node_shared_ptr()->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
        }
    }

    auto dequantized = std::make_shared<ov::opset10::Subtract>(dequantized_weight, dequantized_zero_point);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, gathered_scale);
    auto converted = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f16);
    auto result = std::make_shared<ov::opset10::Result>(converted);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids, weights, zero_point, scale});
}

std::shared_ptr<ov::Model> make_parameter_gather_model(float shift_value) {
    return make_parameter_gather_model(shift_value, shift_value);
}

bool run_host_gather(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& context) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::HostGatherQuantAsymm<>>(std::ref(context));
    return rewrite.run_on_model(model);
}

enum class VocabMatMulTerminal { Add, Transpose, Convert, Gated };

std::shared_ptr<ov::Model> make_vocab_matmul_model(bool convert_before_matmul,
                                                   std::optional<VocabMatMulTerminal> terminal = std::nullopt,
                                                   std::size_t vocab_size = 4) {
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto weights = ov::opset10::Constant::create(ov::element::u8, ov::Shape{vocab_size, 2}, std::vector<uint8_t>(vocab_size * 2, 200));
    auto zero_point =
        ov::opset10::Constant::create(ov::element::u8, ov::Shape{vocab_size, 1}, std::vector<uint8_t>(vocab_size, 128));
    const auto scale_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto scale = ov::opset10::Constant::create(scale_type, ov::Shape{vocab_size, 1}, std::vector<float>(vocab_size, 1.0f));
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, scale_type);
    weight_convert->set_friendly_name("vocab_weight_convert");
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, scale_type);
    zero_point_convert->set_friendly_name("vocab_zero_point_convert");
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    ov::Output<ov::Node> matmul_weights = scaled;
    if (convert_before_matmul) {
        matmul_weights = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    }
    auto matmul = std::make_shared<ov::opset10::MatMul>(hidden, matmul_weights, false, true);
    ov::Output<ov::Node> output = matmul;
    if (terminal == VocabMatMulTerminal::Add) {
        auto bias = ov::opset10::Constant::create(ov::element::f32, ov::Shape{1, vocab_size}, std::vector<float>(vocab_size, 1.0f));
        output = std::make_shared<ov::opset10::Add>(matmul, bias);
    } else if (terminal == VocabMatMulTerminal::Transpose) {
        auto order = ov::opset10::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        output = std::make_shared<ov::opset10::Transpose>(matmul, order);
    } else if (terminal == VocabMatMulTerminal::Convert) {
        output = std::make_shared<ov::opset10::Convert>(matmul, ov::element::f32);
    } else if (terminal == VocabMatMulTerminal::Gated) {
        auto divisor = ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        auto div = std::make_shared<ov::opset10::Divide>(matmul, divisor);
        auto tanh = std::make_shared<ov::opset10::Tanh>(div);
        auto multiplier = ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
        output = std::make_shared<ov::opset10::Multiply>(tanh, multiplier);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{hidden});
}

std::shared_ptr<ov::Model> make_pretransposed_vocab_matmul_model(bool convert_before_matmul, std::size_t vocab_size = 4) {
    auto hidden = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto weights = ov::opset10::Constant::create(ov::element::u8, ov::Shape{2, vocab_size}, std::vector<uint8_t>(vocab_size * 2, 200));
    auto zero_point =
        ov::opset10::Constant::create(ov::element::u8, ov::Shape{1, vocab_size}, std::vector<uint8_t>(vocab_size, 128));
    const auto scale_type = convert_before_matmul ? ov::element::f16 : ov::element::f32;
    auto scale = ov::opset10::Constant::create(scale_type, ov::Shape{1, vocab_size}, std::vector<float>(vocab_size, 1.0f));
    auto weight_convert = std::make_shared<ov::opset10::Convert>(weights, scale_type);
    weight_convert->set_friendly_name("vocab_weight_convert");
    auto zero_point_convert = std::make_shared<ov::opset10::Convert>(zero_point, scale_type);
    zero_point_convert->set_friendly_name("vocab_zero_point_convert");
    auto dequantized = std::make_shared<ov::opset10::Subtract>(weight_convert, zero_point_convert);
    auto scaled = std::make_shared<ov::opset10::Multiply>(dequantized, scale);
    ov::Output<ov::Node> matmul_weights = scaled;
    if (convert_before_matmul) {
        matmul_weights = std::make_shared<ov::opset10::Convert>(scaled, ov::element::f32);
    }
    auto matmul = std::make_shared<ov::opset10::MatMul>(hidden, matmul_weights, false, false);
    return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{hidden});
}

std::size_t count_subtracts(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node)) {
            ++count;
        }
    }
    return count;
}

std::size_t count_sub128_shifts(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node) &&
            node->get_rt_info().count(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO) > 0) {
            ++count;
        }
    }
    return count;
}

void clear_sub128_shift_markers(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::opset10::Subtract>(node)) {
            node->get_rt_info().erase(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO);
        }
    }
}

bool contains_node(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto nodes = model->get_ordered_ops();
    return std::any_of(nodes.begin(), nodes.end(), [&](const auto& node) {
        return node->get_friendly_name() == name;
    });
}

std::map<std::shared_ptr<ov::op::v0::Parameter>, std::shared_ptr<ov::op::v0::Constant>>
parameterize_vocab(const std::shared_ptr<ov::Model>& model) {
    std::map<std::shared_ptr<ov::op::v0::Parameter>, std::shared_ptr<ov::op::v0::Constant>> sources;
    for (const auto& node : model->get_ordered_ops()) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (constant && constant->get_element_type() == ov::element::u8) {
            auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, constant->get_shape());
            ov::replace_node(constant, parameter);
            model->add_parameters({parameter});
            sources.emplace(parameter, constant);
        }
    }
    return sources;
}

bool run_extract(const std::shared_ptr<ov::Model>& model, ov::npuw::patterns::opt::Context& context) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::ExtractVocabSub128>(std::ref(context));
    return rewrite.run_on_model(model);
}

}  // namespace

TEST(DQLiftGatherAsymCWTest, LiftsPairedSub128Shifts) {
    const auto model = make_gather_model(128.0f);

    EXPECT_TRUE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 3);
}

TEST(DQLiftGatherAsymCWTest, RejectsNon128Subtractions) {
    const auto model = make_gather_model(127.0f);

    EXPECT_FALSE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 1);
}

TEST(DQLiftGatherAsymCWTest, RejectsUnmarked128Subtractions) {
    const auto model = make_gather_model(128.0f);
    clear_sub128_shift_markers(model);

    EXPECT_FALSE(run_lift(model));
    EXPECT_EQ(count_gathers(model), 1);
}

TEST(HostGatherQuantAsymmTest, AcceptsPairedSub128Shifts) {
    ov::npuw::patterns::opt::Context context;
    EXPECT_TRUE(run_host_gather(make_parameter_gather_model(128.0f), context));
    ASSERT_TRUE(context.params_to_quant_gather_unpack.has_value());
    ASSERT_EQ(context.params_to_quant_gather_unpack->params_to_runtime_unpack_gather.size(), 1);
}

TEST(HostGatherQuantAsymmTest, RejectsNon128Subtractions) {
    ov::npuw::patterns::opt::Context context;
    EXPECT_FALSE(run_host_gather(make_parameter_gather_model(127.0f), context));
}

TEST(HostGatherQuantAsymmTest, RejectsUnmarked128Subtractions) {
    const auto model = make_parameter_gather_model(128.0f);
    clear_sub128_shift_markers(model);
    ov::npuw::patterns::opt::Context context;

    EXPECT_FALSE(run_host_gather(model, context));
    EXPECT_FALSE(context.params_to_quant_gather_unpack.has_value());
    EXPECT_EQ(count_subtracts(model), 3);
}

using VocabSub128TestParams = std::tuple<bool, bool>;

class InsertVocabSub128PrePostProcessingTest : public ::testing::TestWithParam<VocabSub128TestParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<VocabSub128TestParams>& info) {
        const auto [pretransposed_layout, convert_before_matmul] = info.param;
        return std::string(pretransposed_layout ? "Pretransposed" : "Standard") +
               (convert_before_matmul ? "WithConvert" : "DirectMatMul");
    }
};

TEST_P(InsertVocabSub128PrePostProcessingTest, PreservesVocabularyConverts) {
    const auto [pretransposed_layout, convert_before_matmul] = GetParam();
    const auto model = pretransposed_layout ? make_pretransposed_vocab_matmul_model(convert_before_matmul) :
                                              make_vocab_matmul_model(convert_before_matmul);
    EXPECT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 3u);
    EXPECT_EQ(count_sub128_shifts(model), 2u);

    const auto nodes = model->get_ordered_ops();
    const auto weight_convert = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return node->get_friendly_name() == "vocab_weight_convert";
    });
    const auto zero_point_convert = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return node->get_friendly_name() == "vocab_zero_point_convert";
    });
    ASSERT_NE(weight_convert, nodes.end());
    ASSERT_NE(zero_point_convert, nodes.end());
    EXPECT_TRUE(ov::is_decompression(*weight_convert));
    EXPECT_TRUE(ov::is_decompression(*zero_point_convert));

    ov::preprocess::PrePostProcessor(model).build();

    EXPECT_TRUE(contains_node(model, "vocab_weight_convert"));
    EXPECT_TRUE(contains_node(model, "vocab_zero_point_convert"));
}

TEST_P(InsertVocabSub128PrePostProcessingTest, ExtractsShiftsIntoLazyI8Inputs) {
    const auto [pretransposed_layout, convert_before_matmul] = GetParam();
    const auto model = pretransposed_layout ? make_pretransposed_vocab_matmul_model(convert_before_matmul) :
                                             make_vocab_matmul_model(convert_before_matmul);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    ov::Tensor hidden(ov::element::f32, ov::Shape{1, 2});
    hidden.data<float>()[0] = 1.0f;
    hidden.data<float>()[1] = 2.0f;
    ov::TensorVector expected{ov::Tensor(ov::element::f32, ov::Shape{1, 4})};
    ASSERT_TRUE(model->evaluate(expected, {hidden}));

    const auto sources = parameterize_vocab(model);
    ov::npuw::patterns::opt::Context context;
    ASSERT_TRUE(run_extract(model, context));
    ASSERT_EQ(context.params_to_subtract_128.size(), 2u);
    EXPECT_EQ(count_subtracts(model), 1u);
    EXPECT_EQ(count_sub128_shifts(model), 0u);

    ov::TensorVector inputs{hidden};
    for (std::size_t index = 1; index < model->get_parameters().size(); ++index) {
        inputs.push_back(ov::npuw::weights::LazyTensor(sources.at(model->get_parameters()[index])).eval());
    }
    for (const auto& entry : context.params_to_subtract_128) {
        EXPECT_EQ(entry.first->get_element_type(), ov::element::i8);
        EXPECT_EQ(entry.second->get_element_type(), ov::element::u8);
        model->add_parameters({entry.first});
        const auto source = ov::as_type_ptr<ov::op::v0::Parameter>(entry.second);
        inputs.push_back(ov::npuw::weights::LazyTensor(sources.at(source)).subtract_128().eval());
    }
    model->validate_nodes_and_infer_types();
    ov::TensorVector actual{ov::Tensor(ov::element::f32, ov::Shape{1, 4})};
    ASSERT_TRUE(model->evaluate(actual, inputs));
    for (std::size_t index = 0; index < expected.front().get_size(); ++index) {
        EXPECT_FLOAT_EQ(expected.front().data<float>()[index], actual.front().data<float>()[index]);
    }
    EXPECT_FALSE(run_extract(model, context));
}

TEST_P(InsertVocabSub128PrePostProcessingTest, PartitioningCreatesWeightlessI8Closures) {
    const auto [pretransposed_layout, convert_before_matmul] = GetParam();
    constexpr std::size_t vocab_size = 32;
    const auto model = pretransposed_layout ? make_pretransposed_vocab_matmul_model(convert_before_matmul, vocab_size) :
                                             make_vocab_matmul_model(convert_before_matmul, std::nullopt, vocab_size);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    auto options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*options);
    ::intel_npu::Config config(options);
    config.update({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_FUNCALL_FOR_ALL", "YES"},
                   {"NPUW_FOLD", "YES"}, {"NPUW_DQ", "NO"}, {"NPUW_HOST_GATHER", "NO"}});
    ov::npuw::PartitioningContext context;
    context.use_host_gather_quant = true;
    const auto partitioning = ov::npuw::getPartitioning(model, config, context);
    ASSERT_EQ(partitioning.functions.size(), 1u);
    const auto& function = partitioning.functions.begin()->second;
    EXPECT_EQ(count_subtracts(function._model), 1u);
    EXPECT_EQ(count_sub128_shifts(function._model), 0u);
    ASSERT_EQ(function._param_offset, 1u);
    ASSERT_EQ(function._model->get_parameters().size(), 4u);
    std::size_t checked_calls = 0;
    for (const auto& subgraph : partitioning.subgraphs) {
        if (subgraph._funcall.empty()) {
            continue;
        }
        ++checked_calls;
        ASSERT_EQ(subgraph._lazy_closure.size(), 3u);
        ASSERT_EQ(subgraph._closure.size(), 3u);
        ASSERT_EQ(subgraph._is_lazy_unpack.size(), 3u);
        ov::Tensor hidden(ov::element::f32, ov::Shape{1, 2});
        hidden.data<float>()[0] = 1.0f;
        hidden.data<float>()[1] = 2.0f;
        ov::TensorVector inputs{hidden};
        std::size_t shifted_count = 0;
        for (std::size_t index = 0; index < subgraph._lazy_closure.size(); ++index) {
            const auto& lazy = subgraph._lazy_closure[index];
            const auto& parameter = function._model->get_parameters()[index + function._param_offset];
            EXPECT_EQ(lazy.eval_meta().type, parameter->get_element_type());
            EXPECT_EQ(lazy.eval_meta().shape, parameter->get_shape());
            const auto transforms = lazy.get_transformations();
            if (std::holds_alternative<ov::npuw::weights::op::Subtract128>(transforms.front())) {
                ++shifted_count;
            }
            inputs.push_back(lazy.eval());
        }
        EXPECT_EQ(shifted_count, 2u);
        ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{1, vocab_size})};
        ASSERT_TRUE(function._model->evaluate(outputs, inputs));
        for (std::size_t index = 0; index < vocab_size; ++index) {
            EXPECT_FLOAT_EQ(outputs.front().data<float>()[index], 216.0f);
        }
    }
    EXPECT_EQ(checked_calls, 1u);
}

INSTANTIATE_TEST_SUITE_P(
    LayoutAndConversion,
    InsertVocabSub128PrePostProcessingTest,
    ::testing::Values(std::make_tuple(false, false),
                      std::make_tuple(false, true),
                      std::make_tuple(true, false),
                      std::make_tuple(true, true)),
    InsertVocabSub128PrePostProcessingTest::getTestCaseName);

class InsertVocabSub128LmHeadTerminalTest : public ::testing::TestWithParam<VocabMatMulTerminal> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<VocabMatMulTerminal>& info) {
        switch (info.param) {
        case VocabMatMulTerminal::Add:
            return "Add";
        case VocabMatMulTerminal::Transpose:
            return "Transpose";
        case VocabMatMulTerminal::Convert:
            return "Convert";
        case VocabMatMulTerminal::Gated:
            return "Gated";
        }
        return "Unknown";
    }
};

TEST_P(InsertVocabSub128LmHeadTerminalTest, InsertsSub128BeforeLmHeadTerminal) {
    const auto model = make_vocab_matmul_model(false, GetParam());

    EXPECT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 3u);
    EXPECT_EQ(count_sub128_shifts(model), 2u);
}

INSTANTIATE_TEST_SUITE_P(LmHeadTerminal,
                         InsertVocabSub128LmHeadTerminalTest,
                         ::testing::Values(VocabMatMulTerminal::Add,
                                           VocabMatMulTerminal::Transpose,
                                           VocabMatMulTerminal::Convert,
                                           VocabMatMulTerminal::Gated),
                         InsertVocabSub128LmHeadTerminalTest::getTestCaseName);

TEST(InsertVocabSub128LmHeadTerminalTest, SkipsManuallyAddedOutput) {
    const auto model = make_vocab_matmul_model(false);
    model->get_results().front()->get_rt_info()["manually_added_output"] = true;

    EXPECT_FALSE(ov::npuw::InsertVocabSub128().run_on_model(model));
    EXPECT_EQ(count_subtracts(model), 1u);
    EXPECT_EQ(count_sub128_shifts(model), 0u);
}

TEST(ExtractVocabSub128Test, RejectsUnmarked128Subtractions) {
    const auto model = make_vocab_matmul_model(false);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    const auto sources = parameterize_vocab(model);
    ASSERT_EQ(sources.size(), 2u);
    clear_sub128_shift_markers(model);
    ov::npuw::patterns::opt::Context context;
    EXPECT_FALSE(run_extract(model, context));
    EXPECT_TRUE(context.params_to_subtract_128.empty());
    EXPECT_EQ(count_subtracts(model), 3u);
}

TEST(ExtractVocabSub128Test, RejectsSingleMarkedShift) {
    const auto model = make_vocab_matmul_model(false);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    const auto sources = parameterize_vocab(model);
    ASSERT_EQ(sources.size(), 2u);
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_rt_info().erase(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO)) {
            break;
        }
    }
    ov::npuw::patterns::opt::Context context;
    EXPECT_FALSE(run_extract(model, context));
    EXPECT_TRUE(context.params_to_subtract_128.empty());
    EXPECT_EQ(count_subtracts(model), 3u);
}

TEST(ExtractVocabSub128Test, RejectsMarkedNon128Shift) {
    const auto model = make_vocab_matmul_model(false);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    const auto sources = parameterize_vocab(model);
    ASSERT_EQ(sources.size(), 2u);
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_rt_info().count(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO)) {
            node->input(1).replace_source_output(ov::opset10::Constant::create(ov::element::f32, ov::Shape{}, {127}));
        }
    }
    ov::npuw::patterns::opt::Context context;
    EXPECT_FALSE(run_extract(model, context));
    EXPECT_TRUE(context.params_to_subtract_128.empty());
    EXPECT_EQ(count_subtracts(model), 3u);
}

TEST(ExtractVocabSub128Test, PreservesSharedUnshiftedConvert) {
    const auto model = make_vocab_matmul_model(false);
    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    const auto sources = parameterize_vocab(model);
    ASSERT_EQ(sources.size(), 2u);
    const auto nodes = model->get_ordered_ops();
    const auto convert = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
        return node->get_friendly_name() == "vocab_weight_convert";
    });
    ASSERT_NE(convert, nodes.end());
    const auto unshifted = std::make_shared<ov::opset10::Result>(*convert);
    model->add_results({unshifted});
    const auto source = (*convert)->input_value(0);
    ov::npuw::patterns::opt::Context context;
    ASSERT_TRUE(run_extract(model, context));
    EXPECT_EQ(unshifted->input_value(0).get_node_shared_ptr(), *convert);
    EXPECT_EQ((*convert)->input_value(0), source);
    EXPECT_EQ(source.get_element_type(), ov::element::u8);
    EXPECT_EQ(context.params_to_subtract_128.size(), 2u);
}

TEST(LazySubtract128Test, CoversEveryU8ValueWithoutChangingSource) {
    std::vector<uint8_t> values(256);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<uint8_t>(index);
    }
    const auto constant = ov::opset10::Constant::create(ov::element::u8, ov::Shape{16, 16}, values);
    ov::npuw::weights::LazyTensor source(constant);
    const auto shifted = source.subtract_128();
    EXPECT_EQ(shifted.eval_meta().shape, constant->get_shape());
    EXPECT_EQ(shifted.eval_meta().type, ov::element::i8);
    EXPECT_EQ(shifted, source.subtract_128());
    EXPECT_NE(shifted, source);
    const auto result = shifted.eval();
    for (std::size_t index = 0; index < values.size(); ++index) {
        EXPECT_EQ(result.data<const int8_t>()[index], static_cast<int>(index) - 128);
    }
    EXPECT_EQ(constant->cast_vector<uint8_t>(), values);
}

TEST(LazySubtract128Test, RejectsNonU8Input) {
    const auto constant = ov::opset10::Constant::create(ov::element::i8, ov::Shape{1}, {0});
    const auto shifted = ov::npuw::weights::LazyTensor(constant).subtract_128();
    EXPECT_THROW(shifted.eval(), ov::Exception);
    EXPECT_THROW(shifted.eval_meta(), ov::Exception);
}

