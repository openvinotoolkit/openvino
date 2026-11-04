// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {
namespace {
struct NGramHashCase {
    std::string name;
    std::vector<int64_t> multipliers;
    std::vector<int64_t> heads_per_order;
    int64_t num_engram;
    std::vector<int64_t> vocab_sizes = {};
};

const std::vector<int64_t> qwen_multipliers{23703573157769, 20109073645365, 8052911324071, 3000000019};
const std::vector<int64_t> deepseek_multipliers{0x3BF58476D1CE4E5B,
                                              static_cast<int64_t>(0x9E3779B97F4A7C15ULL),
                                              1234567891011,
                                              static_cast<int64_t>(0xD6E8FEB86659FD93ULL)};

const std::vector<NGramHashCase> cases{
    {"Qwen3_8FlashNext_Bigram", {qwen_multipliers[0], qwen_multipliers[1]}, {8}, 0},
    {"Qwen3_8FlashNext_Trigram", {qwen_multipliers.begin(), qwen_multipliers.begin() + 3}, {8, 8}, 0},
    {"Qwen3_8FlashNext_Fourgram", qwen_multipliers, {8, 8, 8}, 0},
    {"DeepSeekV4_1Flash_Bigram", {deepseek_multipliers[0], deepseek_multipliers[1]}, {4}, 8},
    {"DeepSeekV4_1Flash_Trigram", {deepseek_multipliers.begin(), deepseek_multipliers.begin() + 3}, {4, 4}, 8},
    {"DeepSeekV4_1Flash_Fourgram_PerOrderHeads", deepseek_multipliers, {2, 4, 3}, 8},
    {"Barrett_UnitAndPowerOfTwo",
     {0, std::numeric_limits<int64_t>::min(), -1, std::numeric_limits<int64_t>::max()},
     {0, 2, 1},
     0,
     {1, 2, 1073741824}},
    {"Barrett_MaxDivisor",
     {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()},
      {2},
     0,
      {std::numeric_limits<int32_t>::max(), 1}},
    {"Barrett_NearPowerOfTwo",
     {qwen_multipliers.begin(), qwen_multipliers.begin() + 3},
     {2, 1},
     0,
     {536870911, 536870913, 536870923}},
};

enum class HashInputKind { direct_i32, direct_i64, traced_i64 };
using NGramHashParams = std::tuple<NGramHashCase, HashInputKind>;

std::vector<int64_t> make_vocab_sizes(const NGramHashCase& config) {
    if (!config.vocab_sizes.empty()) {
        return config.vocab_sizes;
    }
    std::vector<int64_t> vocab;
    for (const auto heads : config.heads_per_order) {
        for (int64_t head = 0; head < heads; ++head) {
            vocab.push_back(20000003 + static_cast<int64_t>(vocab.size()) * 20);
        }
    }
    return vocab;
}

std::vector<int64_t> make_offsets(const std::vector<int64_t>& vocab) {
    std::vector<int64_t> offsets;
    int64_t offset = 0;
    for (const auto size : vocab) {
        offsets.push_back(offset);
        offset += size;
    }
    return offsets;
}

std::shared_ptr<Node> make_hash(const Output<Node>& windows, const NGramHashCase& config) {
    const auto vocab = make_vocab_sizes(config);
    return std::make_shared<op::internal::NGramHashFunction>(windows,
                                                            config.multipliers.size(),
                                                            config.heads_per_order,
                                                            config.multipliers,
                                                            vocab,
                                                            make_offsets(vocab));
}

std::shared_ptr<Model> make_direct_model(const NGramHashCase& config, const element::Type& type) {
    PartialShape shape{-1};
    if (config.num_engram) {
        shape.push_back(config.num_engram);
    }
    shape.push_back(config.multipliers.size());
    const auto windows = std::make_shared<op::v0::Parameter>(type, shape);
    windows->set_friendly_name("ngrams");
    return std::make_shared<Model>(OutputVector{make_hash(windows, config)}, ParameterVector{windows});
}

std::shared_ptr<Model> make_traced_model(const NGramHashCase& config, bool reference) {
    PartialShape shape{-1};
    if (config.num_engram) {
        shape.push_back(config.num_engram);
    }
    ParameterVector tokens;
    for (size_t back = 0; back < config.multipliers.size(); ++back) {
        const auto token = std::make_shared<op::v0::Parameter>(element::i64, shape);
        token->set_friendly_name("token_back_" + std::to_string(back));
        tokens.push_back(token);
    }
    const auto axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    if (reference) {
        OutputVector window;
        for (auto token = tokens.rbegin(); token != tokens.rend(); ++token) {
            window.push_back(std::make_shared<op::v0::Unsqueeze>(*token, axis));
        }
        const auto windows = std::make_shared<op::v0::Concat>(window, -1);
        const auto output = std::make_shared<op::v0::Convert>(make_hash(windows, config), element::i64);
        return std::make_shared<Model>(OutputVector{output}, tokens);
    }
    const auto vocab = make_vocab_sizes(config);
    const auto offsets = make_offsets(vocab);
    const auto term = [&](size_t back) {
        return std::make_shared<op::v1::Multiply>(
            tokens[back],
            op::v0::Constant::create(element::i64, Shape{}, {config.multipliers[back]}));
    };
    Output<Node> mix = term(0);
    OutputVector blocks;
    size_t head_offset = 0;
    for (size_t back = 1; back < config.multipliers.size(); ++back) {
        mix = std::make_shared<op::v13::BitwiseXor>(mix, term(back));
        const auto heads = static_cast<size_t>(config.heads_per_order[back - 1]);
        if (heads == 0) {
            continue;
        }
        const std::vector<int64_t> sizes(vocab.begin() + head_offset, vocab.begin() + head_offset + heads);
        const std::vector<int64_t> starts(offsets.begin() + head_offset, offsets.begin() + head_offset + heads);
        head_offset += heads;
        const auto expanded = std::make_shared<op::v0::Unsqueeze>(mix, axis);
        const auto mod = std::make_shared<op::v1::FloorMod>(
            expanded,
            op::v0::Constant::create(element::i64, Shape{heads}, sizes));
        blocks.push_back(std::make_shared<op::v1::Add>(
            mod,
            op::v0::Constant::create(element::i64, Shape{heads}, starts)));
    }
    const Output<Node> output = blocks.size() == 1 ? blocks[0] : std::make_shared<op::v0::Concat>(blocks, -1)->output(0);
    return std::make_shared<Model>(OutputVector{output}, tokens);
}
}  // namespace

class NGramHashFunctionCPUTest : public testing::WithParamInterface<NGramHashParams>, public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NGramHashParams>& info) {
        const auto& [config, kind] = info.param;
        const std::string mode = kind == HashInputKind::traced_i64 ? "TracedI64"
                                : kind == HashInputKind::direct_i64 ? "DirectI64"
                                                                    : "DirectI32";
        return config.name + "_" + mode;
    }

protected:
    void SetUp() override {
        const auto& [config, kind] = GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto type = kind == HashInputKind::direct_i32 ? element::i32 : element::i64;
        if (kind == HashInputKind::traced_i64) {
            function = make_traced_model(config, false);
            functionRefs = make_traced_model(config, true);
        } else {
            function = make_direct_model(config, type);
        }
        std::vector<InputShape> shapes;
        for (const auto& parameter : function->get_parameters()) {
            std::vector<Shape> static_shapes;
            for (const size_t tokens : {1, 3, 257, 2}) {
                Shape shape{tokens};
                if (config.num_engram) {
                    shape.push_back(config.num_engram);
                }
                if (kind != HashInputKind::traced_i64) {
                    shape.push_back(config.multipliers.size());
                }
                static_shapes.push_back(shape);
            }
            shapes.emplace_back(parameter->get_partial_shape(), static_shapes);
        }
        init_input_shapes(shapes);
    }

    void generate_inputs(const std::vector<Shape>& shapes) override {
        inputs.clear();
        std::mt19937 generator(static_cast<uint32_t>(shape_size(shapes[0])));
        const auto low = std::get<1>(GetParam()) == HashInputKind::traced_i64 ? 0 : std::numeric_limits<int32_t>::min();
        std::uniform_int_distribution<int64_t> distribution(low, std::numeric_limits<int32_t>::max());
        const auto& parameters = function->get_parameters();
        for (size_t input = 0; input < parameters.size(); ++input) {
            Tensor tensor(parameters[input]->get_element_type(), shapes[input]);
            for (size_t index = 0; index < tensor.get_size(); ++index) {
                const auto value = distribution(generator);
                if (tensor.get_element_type() == element::i32) {
                    tensor.data<int32_t>()[index] = static_cast<int32_t>(value);
                } else {
                    tensor.data<int64_t>()[index] = value;
                }
            }
            inputs.emplace(parameters[input], tensor);
        }
    }

    void init_thresholds() override {
        abs_threshold = 0;
        rel_threshold = 0;
    }
};

TEST_P(NGramHashFunctionCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "NgramHash", 1);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGramHashFunction,
                         NGramHashFunctionCPUTest,
                         testing::Combine(testing::ValuesIn(cases),
                                          testing::Values(HashInputKind::direct_i32,
                                                          HashInputKind::direct_i64,
                                                          HashInputKind::traced_i64)),
                         NGramHashFunctionCPUTest::getTestCaseName);
}  // namespace ov::test