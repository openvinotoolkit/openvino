// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "subgraphs_builders.hpp"

using namespace ov::test;

namespace SubgraphTestsDefinitions {

using KVCacheTestParams = std::tuple<std::vector<InputShape>,  // input shapes
                                     ov::element::Type,        // in/out precision
                                     std::map<std::string, std::string>>;  // additional config

class KVCacheTest : public testing::WithParamInterface<KVCacheTestParams>, public SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<KVCacheTestParams> obj) {
        std::vector<InputShape> input_shapes;
        ov::element::Type element_type;
        std::map<std::string, std::string> additional_config;

        std::tie(input_shapes, element_type, additional_config) = obj.param;

        std::ostringstream result;
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "precision=" << element_type << "_";
        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        ov::element::Type element_type;
        std::map<std::string, std::string> additional_config;

        std::tie(input_shapes, element_type, additional_config) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(input_shapes);

        inType = outType = element_type;

        function = tests::make_llm_kv_cache_pattern(inputDynamicShapes[0][0], inputDynamicShapes[0][1], inputDynamicShapes[0][3], element_type);
    }
};

TEST_P(KVCacheTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {
        {{-1, 32, -1, 80}, { {1, 32, 0, 80}, {1, 32, 20, 80} }},
        {{-1, -1, 32, 80}, { {1, 20, 32, 80}, {1, 1, 32, 80} }},
        {{-1, 32, -1, -1}, { {1, 32, 1, 20}, {1, 32, 1, 21} }}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_Dynamic,
                         KVCacheTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(precisions),
                                            ::testing::Values(std::map<std::string, std::string>())),
                         KVCacheTest::get_test_case_name);
} // namespace

TEST(KVCacheTest, smoke_multipleIterations) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();

    const size_t batch = 1;
    const size_t n_heads = 32;
    const size_t n_features = 80;
    const size_t context_size = 20;
    size_t cache_size = 0;

    ov::element::Type element_type = ov::element::f16;

    auto model = tests::make_llm_kv_cache_pattern(batch, n_heads, n_features, element_type);
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));

    auto input0 = model->get_parameters().at(0);
    auto input1 = model->get_parameters().at(1);
    auto input2 = model->get_parameters().at(2);
    auto output0 = model->get_results().at(0);
    auto output1 = model->get_results().at(1);

    auto get_ref_results = [&model, &input0, &input1, &input2](const ov::Tensor& kv_cache, const ov::Tensor& new_token_data, const ov::Tensor& matmul_data) {
        auto ref_model = model->clone();
        ov::Tensor kv_cache_copy(kv_cache.get_element_type(), kv_cache.get_shape());
        kv_cache.copy_to(kv_cache_copy);
        ngraph::helpers::resize_function(ref_model, {kv_cache_copy.get_shape(), new_token_data.get_shape(), matmul_data.get_shape()});
        return ngraph::helpers::interpretFunction(ref_model, {{input0, kv_cache_copy}, {input1, new_token_data}, {input2, matmul_data}});
    };

    auto compare_tensors = [&model](const std::vector<ov::Tensor> expected, const std::vector<ov::Tensor>& actual) {
            ASSERT_EQ(expected.size(), actual.size());
            ASSERT_EQ(expected.size(), model->get_results().size());
            auto compareMap = ov::test::utils::getCompareMap();
            const auto& results = model->get_results();
            for (size_t j = 0; j < results.size(); j++) {
                const auto result = results[j];
                for (size_t i = 0; i < result->get_input_size(); ++i) {
                    std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
                    if (std::dynamic_pointer_cast<ov::op::v0::Convert>(inputNode)) {
                        std::shared_ptr<ov::Node> nextNodePtr = inputNode->get_input_node_shared_ptr(0);
                        if (!ngraph::is_type<ov::op::v0::Result>(nextNodePtr)) {
                            inputNode = nextNodePtr;
                        }
                    }
                    auto it = compareMap.find(inputNode->get_type_info());
                    ASSERT_NE(it, compareMap.end());
                    it->second(inputNode, i, expected[j], actual[j], 1e-4f, 1e-4f);
                }
            }
    };

    auto infer_request = compiled_model.create_infer_request();
    auto kv_cache_input = infer_request.get_tensor(output0);
    auto matmul_out = infer_request.get_tensor(output1);
    auto new_token_input = infer_request.get_tensor(input1);
    auto matmul_input = infer_request.get_tensor(input2);

    infer_request.set_tensor(input0, kv_cache_input);
    infer_request.set_tensor(input1, new_token_input);
    infer_request.set_tensor(input2, matmul_input);

    {
        const ov::Shape new_token_size_initial = {batch, context_size, n_heads, n_features};
        const ov::Shape kv_cache_size_initial = {batch, n_heads, cache_size, n_features};
        const ov::Shape matmul_in_size_initial = {batch, n_heads, context_size, context_size};

        auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size_initial);
        auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_initial);

        kv_cache_input.set_shape(kv_cache_size_initial);
        new_token_input.set_shape(new_token_data.get_shape());
        matmul_input.set_shape(matmul_data.get_shape());

        new_token_data.copy_to(new_token_input);
        matmul_data.copy_to(matmul_input);

        auto ref_results = get_ref_results(kv_cache_input, new_token_data, matmul_data);

        infer_request.infer();

        compare_tensors(ref_results, {kv_cache_input, matmul_out});

        cache_size += context_size;
    }

    const size_t input_tokens = 1;
    const size_t niters = 10;
    const ov::Shape new_token_size = {batch, input_tokens, n_heads, n_features};
    size_t context_length = cache_size + input_tokens;
    for (size_t i = 0; i < niters; i++, context_length += input_tokens) {
        ov::Shape matmul_in_size_loop = {batch, n_heads, input_tokens, context_length};
        auto new_token_data = ov::test::utils::create_and_fill_tensor(element_type, new_token_size);
        auto matmul_data = ov::test::utils::create_and_fill_tensor(element_type, matmul_in_size_loop);
        auto ref_results = get_ref_results(kv_cache_input, new_token_data, matmul_data);

        new_token_input.set_shape(new_token_data.get_shape());
        matmul_input.set_shape(matmul_data.get_shape());
        new_token_data.copy_to(new_token_input);
        matmul_data.copy_to(matmul_input);

        infer_request.infer();

        compare_tensors(ref_results, {kv_cache_input, matmul_out});
    }
}

} // namespace SubgraphTestsDefinitions
