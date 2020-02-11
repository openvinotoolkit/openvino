// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"

#include <algorithm>
#include <vector>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct unique_test_params {
    std::string model;

    std::string precision;

    std::string sorted;
    std::string return_inverse;
    std::string return_counts;

    InferenceEngine::SizeVector input_shape;
    std::vector<float> input_value;

    InferenceEngine::SizeVector output_uniques_shape;
    InferenceEngine::SizeVector output_indices_shape;
    InferenceEngine::SizeVector output_counts_shape;

    std::vector<float> output_uniques_value_ref;
    std::vector<float> output_indices_value_ref;
    std::vector<float> output_counts_value_ref;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtUniqueTests : public TestsCommon, public WithParamInterface<unique_test_params> {
    std::string getModel(unique_test_params p) {
        std::string model = p.model;

        std::string input_shape;
        std::string output_uniques_shape;
        std::string output_indices_shape;
        std::string output_counts_shape;

        for (auto& shape : p.input_shape) {
            input_shape += "<dim>";
            input_shape += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_uniques_shape) {
            output_uniques_shape += "<dim>";
            output_uniques_shape += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_indices_shape) {
            output_indices_shape += "<dim>";
            output_indices_shape += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_counts_shape) {
            output_counts_shape += "<dim>";
            output_counts_shape += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_SORTED_", p.sorted);
        REPLACE_WITH_STR(model, "_INPUT_SHAPE_", input_shape);
        REPLACE_WITH_STR(model, "_OUTPUT_UNIQUES_SHAPE_", output_uniques_shape);
        REPLACE_WITH_STR(model, "_OUTPUT_INDICES_SHAPE_", output_indices_shape);
        REPLACE_WITH_STR(model, "_OUTPUT_COUNTS_SHAPE_", output_counts_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            unique_test_params p = ::testing::WithParamInterface<unique_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "Unique") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                        node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            // prepare input blob and input blob map
            InferenceEngine::Blob::Ptr input = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_shape) });
            input->allocate();
            auto *input_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input.get());
            std::copy(p.input_value.begin(), p.input_value.end(), (float *)input_ptr->data());
            InferenceEngine::BlobMap input_blob_map;
            input_blob_map["InputValues"] = input;

            // prepare output blob map
            InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap output_blob_map;
            for (auto iter = out.begin(); iter != out.end(); iter++) {
                std::pair<std::string, InferenceEngine::DataPtr> item = *iter;
                InferenceEngine::Blob::Ptr output_blob_ptr = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output_blob_ptr->allocate();
                output_blob_map[item.first] = output_blob_ptr;
            }

            // prepare blobs with reference data
            InferenceEngine::Blob::Ptr output_uniques_blob_ref = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                 p.output_uniques_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_uniques_shape) });
            output_uniques_blob_ref->allocate();
            auto *output_uniques_blob_ref_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(output_uniques_blob_ref.get());
            std::copy(p.output_uniques_value_ref.begin(), p.output_uniques_value_ref.end(), (float *)output_uniques_blob_ref_ptr->data());

            InferenceEngine::Blob::Ptr output_indices_blob_ref = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                 p.output_indices_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_indices_shape) });
            output_indices_blob_ref->allocate();
            auto *output_indices_blob_ref_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(output_indices_blob_ref.get());
            std::copy(p.output_indices_value_ref.begin(), p.output_indices_value_ref.end(), (float *)output_indices_blob_ref_ptr->data());

            InferenceEngine::Blob::Ptr output_counts_blob_ref = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                 p.output_counts_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_counts_shape) });
            output_counts_blob_ref->allocate();
            auto *output_counts_blob_ref_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(output_counts_blob_ref.get());
            std::copy(p.output_counts_value_ref.begin(), p.output_counts_value_ref.end(), (float *)output_counts_blob_ref_ptr->data());

            // infer
            graph.Infer(input_blob_map, output_blob_map);

            // check the result
            auto iter = out.begin();
            compare(*output_blob_map[iter->first], *output_uniques_blob_ref, 0.0f);
            if (p.return_inverse == "true") {
                iter++;
                compare(*output_blob_map[iter->first], *output_indices_blob_ref, 0.0f);
            }
            if (p.return_counts == "true") {
                iter++;
                compare(*output_blob_map[iter->first], *output_counts_blob_ref, 0.0f);
            }
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtUniqueTests, TestsUnique) {}

// model 1 that contains one Unique layer with two outputs: unique elements, indices
std::string model1 = R"V0G0N(
<net Name="Unique_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputValues" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="UniqueLayer" id="1" type="Unique" precision="FP32">
            <data return_counts="false" return_inverse="true" sorted="_SORTED_"/>
            <input>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </input>
            <output>
                <port id="0">
                    _OUTPUT_UNIQUES_SHAPE_
                </port>
                <port id="1">
                    _OUTPUT_INDICES_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

// model 2 that contains one Unique layer with three outputs: unique elements, indices, counts
std::string model2 = R"V0G0N(
<net Name="Unique_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputValues" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="UniqueLayer" id="1" type="Unique" precision="FP32">
            <data return_counts="true" return_inverse="true" sorted="_SORTED_"/>
            <input>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </input>
            <output>
                <port id="0">
                    _OUTPUT_UNIQUES_SHAPE_
                </port>
                <port id="1">
                    _OUTPUT_INDICES_SHAPE_
                </port>
                <port id="2">
                    _OUTPUT_COUNTS_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

// case 1 - input with 10 elements where some of them repeat, non-sorted
InferenceEngine::SizeVector input_shape_case1 = { 10 };
std::vector<float>          input_value_case1 = { 8.f, 1.f, 2.f, 1.f, 8.f, 5.f, 1.f, 5.f, 0.f, 0.f };
InferenceEngine::SizeVector output_uniques_shape_case1 = { 10 };
InferenceEngine::SizeVector output_indicess_shape_case1 = { 10 };
InferenceEngine::SizeVector output_counts_shape_case1 = { 10 };
std::vector<float>          output_uniques_value_ref_case1 = { 8.f, 1.f, 2.f, 5.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
std::vector<float>          output_indices_value_ref_case1 = { 0.f, 1.f, 2.f, 1.f, 0.f, 3.f, 1.f, 3.f, 4.f, 4.f };
std::vector<float>          output_counts_value_ref_case1 = { 2.f, 3.f, 1.f, 2.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f };

// case 2 - input with 10 elements where all of them are unique, non-sorted
InferenceEngine::SizeVector input_shape_case2 = { 10 };
std::vector<float>          input_value_case2 = { 8.f, 1.f, 2.f, 3.f, 10.f, 5.f, 12.f, 15.f, 0.f, 100.f };
InferenceEngine::SizeVector output_uniques_shape_case2 = { 10 };
InferenceEngine::SizeVector output_indicess_shape_case2 = { 10 };
InferenceEngine::SizeVector output_counts_shape_case2 = { 10 };
std::vector<float>          output_uniques_value_ref_case2 = { 8.f, 1.f, 2.f, 3.f, 10.f, 5.f, 12.f, 15.f, 0.f, 100.f };
std::vector<float>          output_indices_value_ref_case2 = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
std::vector<float>          output_counts_value_ref_case2 = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };

// case 3 - input with 10 elements where all of them are the same, non-sorted
InferenceEngine::SizeVector input_shape_case3 = { 10 };
std::vector<float>          input_value_case3 = { 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f };
InferenceEngine::SizeVector output_uniques_shape_case3 = { 10 };
InferenceEngine::SizeVector output_indicess_shape_case3 = { 10 };
InferenceEngine::SizeVector output_counts_shape_case3 = { 10 };
std::vector<float>          output_uniques_value_ref_case3 = { 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f };
std::vector<float>          output_indices_value_ref_case3 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
std::vector<float>          output_counts_value_ref_case3 = { 10.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

// case 4 - input with 10 elements where some of them repeat, sorted
InferenceEngine::SizeVector input_shape_case4 = { 10 };
std::vector<float>          input_value_case4 = { 8.f, 1.f, 2.f, 1.f, 8.f, 5.f, 1.f, 5.f, 0.f, 0.f };
InferenceEngine::SizeVector output_uniques_shape_case4 = { 10 };
InferenceEngine::SizeVector output_indicess_shape_case4 = { 10 };
InferenceEngine::SizeVector output_counts_shape_case4 = { 10 };
std::vector<float>          output_uniques_value_ref_case4 = { 0.f, 1.f, 2.f, 5.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f };
std::vector<float>          output_indices_value_ref_case4 = { 4.f, 1.f, 2.f, 1.f, 4.f, 3.f, 1.f, 3.f, 0.f, 0.f };
std::vector<float>          output_counts_value_ref_case4 = { 2.f, 3.f, 1.f, 2.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f };

// case 5 - input with 10 elements where all of them are unique, sorted
InferenceEngine::SizeVector input_shape_case5 = { 10 };
std::vector<float>          input_value_case5 = { 8.f, 1.f, 2.f, 3.f, 10.f, 5.f, 12.f, 15.f, 0.f, 100.f };
InferenceEngine::SizeVector output_uniques_shape_case5 = { 10 };
InferenceEngine::SizeVector output_indicess_shape_case5 = { 10 };
InferenceEngine::SizeVector output_counts_shape_case5 = { 10 };
std::vector<float>          output_uniques_value_ref_case5 = { 0.f, 1.f, 2.f, 3.f, 5.f, 8.f, 10.f, 12.f, 15.f, 100.f };
std::vector<float>          output_indices_value_ref_case5 = { 5.f, 1.f, 2.f, 3.f, 6.f, 4.f, 7.f, 8.f, 0.f, 9.f };
std::vector<float>          output_counts_value_ref_case5 = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };

INSTANTIATE_TEST_CASE_P(
    TestsUnique, MKLDNNCPUExtUniqueTests,
    ::testing::Values(
        // case 0 - model1, sorted="false", input with 10 elements where some of them repeat
        unique_test_params {
            model1, "FP32", "false", "true", "false", input_shape_case1, input_value_case1,
            output_uniques_shape_case1, output_indicess_shape_case1, output_counts_shape_case1,
            output_uniques_value_ref_case1, output_indices_value_ref_case1, output_counts_value_ref_case1,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 1 - model1, sorted="false", input with 10 elements where all of them are unique
        unique_test_params{
            model1, "FP32", "false", "true", "false", input_shape_case2, input_value_case2,
            output_uniques_shape_case2, output_indicess_shape_case2, output_counts_shape_case2,
            output_uniques_value_ref_case2, output_indices_value_ref_case2, output_counts_value_ref_case2,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 2 - model1, sorted="false", input with 10 elements where all of them are the same
        unique_test_params{
            model1, "FP32", "false", "true", "false", input_shape_case3, input_value_case3,
            output_uniques_shape_case3, output_indicess_shape_case3, output_counts_shape_case3,
            output_uniques_value_ref_case3, output_indices_value_ref_case3, output_counts_value_ref_case3,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 3 - model1, sorted="true", input with 10 elements where some of them repeat
        unique_test_params{
            model1, "FP32", "true", "true", "false", input_shape_case4, input_value_case4,
            output_uniques_shape_case4, output_indicess_shape_case4, output_counts_shape_case4,
            output_uniques_value_ref_case4, output_indices_value_ref_case4, output_counts_value_ref_case4,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 4 - model1, sorted="true", input with 10 elements where all of them are unique
        unique_test_params{
            model1, "FP32", "true", "true", "false", input_shape_case5, input_value_case5,
            output_uniques_shape_case5, output_indicess_shape_case5, output_counts_shape_case5,
            output_uniques_value_ref_case5, output_indices_value_ref_case5, output_counts_value_ref_case5,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 5 - model2, sorted="false", input with 10 elements where some of them repeat
        unique_test_params{
            model2, "FP32", "false", "true", "true", input_shape_case1, input_value_case1,
            output_uniques_shape_case1, output_indicess_shape_case1, output_counts_shape_case1,
            output_uniques_value_ref_case1, output_indices_value_ref_case1, output_counts_value_ref_case1,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 6 - model2, sorted="false", input with 10 elements where all of them are unique
        unique_test_params{
            model2, "FP32", "false", "true", "true", input_shape_case2, input_value_case2,
            output_uniques_shape_case2, output_indicess_shape_case2, output_counts_shape_case2,
            output_uniques_value_ref_case2, output_indices_value_ref_case2, output_counts_value_ref_case2,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 7 - model2, sorted="false", input with 10 elements where all of them are the same
        unique_test_params{
            model2, "FP32", "false", "true", "true", input_shape_case3, input_value_case3,
            output_uniques_shape_case3, output_indicess_shape_case3, output_counts_shape_case3,
            output_uniques_value_ref_case3, output_indices_value_ref_case3, output_counts_value_ref_case3,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 8 - model2, sorted="true", input with 10 elements where some of them repeat
        unique_test_params{
            model2, "FP32", "true", "true", "true", input_shape_case4, input_value_case4,
            output_uniques_shape_case4, output_indicess_shape_case4, output_counts_shape_case4,
            output_uniques_value_ref_case4, output_indices_value_ref_case4, output_counts_value_ref_case4,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 9 - model2, sorted="true", input with 10 elements where all of them are unique
        unique_test_params{
            model2, "FP32", "true", "true", "true", input_shape_case5, input_value_case5,
            output_uniques_shape_case5, output_indicess_shape_case5, output_counts_shape_case5,
            output_uniques_value_ref_case5, output_indices_value_ref_case5, output_counts_value_ref_case5,
            1, MKLDNNPlugin::impl_desc_type::unknown
        }
));
