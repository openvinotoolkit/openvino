// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <map>
#include <string>

#include <mixed_affinity_functions.hpp>
#include "ngraph_transformations/mixed_affinity.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"


using namespace ov::intel_cpu;
using namespace ov::intel_cpu::mixed_affinity;
struct SubgraphInOut {
    std::string name;
    size_t idx;
};

using SubgraphInputs = std::vector<SubgraphInOut>;
using SubgraphOutputs = SubgraphInputs;
using SubgraphsMarkup = std::unordered_map<Characteristics, std::pair<SubgraphInputs, SubgraphOutputs>>;

class FormSubgraphsTests: public TransformationTests {
public:
    void TearDown() override {
        OPENVINO_ASSERT(model != nullptr, "Test Model is not initialized.");
        const auto subgraphs = MixedAffinity::formSubgraphs(model);
        ASSERT_EQ(subgraphs.size(), ref_subgraphs.size());

        for (const auto& ref_subgraph : ref_subgraphs) {
            const auto& act_subgraph = subgraphs.find(ref_subgraph.first);
            ASSERT_NE(act_subgraph, subgraphs.end()) << "Expected subgraph wasn't found.";

            const auto& ref_inputs = ref_subgraph.second.first;
            const auto& act_inputs = act_subgraph->second.starts;
            ASSERT_EQ(act_inputs.size(), ref_inputs.size());
            for (size_t i = 0; i < ref_inputs.size(); ++i) {
                const auto act_name = act_inputs[i].get_node()->get_friendly_name();
                ASSERT_EQ(act_name, ref_inputs[i].name);
                const auto act_idx = act_inputs[i].get_index();
                ASSERT_EQ(act_idx, ref_inputs[i].idx);
            }

            const auto& ref_outputs = ref_subgraph.second.first;
            const auto& act_outputs = act_subgraph->second.starts;
            ASSERT_EQ(act_outputs.size(), ref_outputs.size());
            for (size_t i = 0; i < ref_outputs.size(); ++i) {
                const auto act_name = act_outputs[i].get_node()->get_friendly_name();
                ASSERT_EQ(act_name, ref_outputs[i].name);
                const auto act_idx = act_outputs[i].get_index();
                ASSERT_EQ(act_idx, ref_outputs[i].idx);
            }
        }
    }

    std::shared_ptr<ov::Model> model;
    SubgraphsMarkup ref_subgraphs;
};


TEST_F(FormSubgraphsTests, ConvWithBias) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithBiasFunction builder({input_shape});
    MixedAffinityMarkup markup{{"convolution", {1, 4}}, {"bias", {1, 4}}};
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution", 0}}, SubgraphOutputs{{"bias", 0}}}
        }
    };
}


TEST_F(FormSubgraphsTests, ConvWithBias2) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithBiasFunction builder({input_shape});
    MixedAffinityMarkup markup{{"convolution", {1, 4}}, {"bias", {2, 2}}};
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution", 0}}, SubgraphOutputs{{"convolution", 0}}}
        },
        {
            Characteristics{2, 2},
            {SubgraphInputs{{"bias", 0}}, SubgraphOutputs{{"bias", 0}}}
        }
    };
}

TEST_F(FormSubgraphsTests, ConvWithTranspose) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithTransposeFunction builder({input_shape});
    MixedAffinityMarkup markup{{"convolution", {1, 4}}, {"transpose", {1, 4}}};
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution", 0}}, SubgraphOutputs{{"transpose", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, ConvWithReshapeDynamicShapes) {
    ov::PartialShape input_shape{4, 3, -1, -1};
    ConvWithReshapeFunction builder({input_shape});
    MixedAffinityMarkup markup{{"convolution", {1, 4}}};
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution", 0}}, SubgraphOutputs{{"convolution", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, TwoConvAndAddEqualShapes) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape, input_shape});
    MixedAffinityMarkup markup{
        {"convolution_1", {1, 4}},
        {"bias_1", {1, 4}},
        {"convolution_2", {1, 4}},
        {"bias_2", {1, 4}},
        {"add", {1, 4}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution_1", 0}, {"convolution_2", 0}}, SubgraphOutputs{{"add", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, TwoConvAndAddDifferentShapes) {
    ov::PartialShape input_shape_1{4, 3, 16, 16};
    ov::PartialShape input_shape_2{1, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape_1, input_shape_2});
    MixedAffinityMarkup markup{
        {"convolution_1", {1, 4}},
        {"bias_1", {1, 4}},
        {"convolution_2", {1, 1}},
        {"bias_2", {1, 1}},
        {"add", {1, 4}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution_1", 0}, {"add", 1}}, SubgraphOutputs{{"add", 0}}}
        },
        {
            Characteristics{1, 1},
            {SubgraphInputs{{"convolution_2", 0}}, SubgraphOutputs{{"bias_2", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, TwoConvAndAddDifferentOptBatches) {
    ov::PartialShape input_shape_1{4, 3, 16, 16};
    ov::PartialShape input_shape_2{1, 3, 16, 16};
    TwoConvAndAddFunction builder({input_shape_1, input_shape_2});
    MixedAffinityMarkup markup{
        {"convolution_1", {2, 2}},
        {"bias_1", {2, 2}},
        {"convolution_2", {1, 1}},
        {"bias_2", {1, 1}},
        {"add", {2, 2}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{2, 2},
            {SubgraphInputs{{"convolution_1", 0}, {"add", 1}}, SubgraphOutputs{{"add", 0}}}
        },
        {
            Characteristics{1, 1},
            {SubgraphInputs{{"convolution_2", 0}}, SubgraphOutputs{{"bias_2", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, TwoConvWithS2BFunction) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    TwoConvWithS2BFunction builder({input_shape});
    MixedAffinityMarkup markup{
        {"convolution_1", {1, 4}},
        {"convolution_2", {1, 16}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution_1", 0}}, SubgraphOutputs{{"convolution_1", 0}}}
        },
        {
            Characteristics{1, 16},
            {SubgraphInputs{{"convolution_2", 0}}, SubgraphOutputs{{"convolution_2", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, ConvWithSplitAndResultFunction) {
    ov::PartialShape input_shape{4, 3, 16, 16};
    ConvWithSplitAndResultFunction builder({input_shape});
    MixedAffinityMarkup markup{
        {"convolution_1", {1, 4}},
        {"relu_1", {1, 4}},
        {"split", {1, 4}},
        {"convolution_2", {2, 2}},
        {"relu_2", {2, 2}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 4},
            {SubgraphInputs{{"convolution_1", 0}}, SubgraphOutputs{{"split", 0}, {"split", 1}}}
        },
        {
            Characteristics{2, 2},
            {SubgraphInputs{{"convolution_2", 0}}, SubgraphOutputs{{"relu_2", 0}}}
        },
    };
}

TEST_F(FormSubgraphsTests, GrConvWithParamFunction) {
    ov::PartialShape input_shape{8, 3, 56, 56};
    ov::PartialShape weights_shape{3, 1, 1, 3, 3};
    GrConvWithParamFunction builder({input_shape, weights_shape});
    MixedAffinityMarkup markup{
        {"group_conv", {1, 8}},
    };
    model = builder.getOriginal(markup);

    ref_subgraphs = {
        {
            Characteristics{1, 8},
            {SubgraphInputs{{"group_conv", 0}, {"group_conv", 1}}, SubgraphOutputs{{"group_conv", 0}}}
        },
    };
}
