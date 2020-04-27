// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"
#include <ie_core.hpp>

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"

#include <algorithm>
#include <vector>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct bucketize_test_params {
    std::string model;
    std::string precision;
    std::string right;

    InferenceEngine::SizeVector input_shape;
    std::vector<float> input_value;

    bool with_second_input;
    InferenceEngine::SizeVector boundaries_shape;
    std::vector<float> boundaries_value;

    InferenceEngine::SizeVector output_shape;
    std::vector<int> output_value_ref;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtBucketizeTests : public TestsCommon, public WithParamInterface<bucketize_test_params> {
    std::string getModel(bucketize_test_params p) {
        std::string model = p.model;

        std::string input_shape;
        std::string boundaries_shape;
        std::string output_shape;

        for (auto& shape : p.input_shape) {
            input_shape += "<dim>";
            input_shape += std::to_string(shape) + "</dim>\n";
        }
        if (p.with_second_input) {
            for (auto& shape : p.boundaries_shape) {
                boundaries_shape += "<dim>";
                boundaries_shape += std::to_string(shape) + "</dim>\n";
            }
        }

        for (auto& shape : p.output_shape) {
            output_shape += "<dim>";
            output_shape += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_RIGHT_", p.right);
        REPLACE_WITH_STR(model, "_INPUT_SHAPE_", input_shape);
        REPLACE_WITH_STR(model, "_BOUNDARIES_SHAPE_", boundaries_shape);
        REPLACE_WITH_STR(model, "_OUTPUT_SHAPE_", output_shape);

        return model;
    }

protected:
    static void compare_int(
        InferenceEngine::Blob &res,
        InferenceEngine::Blob &ref,
        int max_diff = 0,
        const std::string assertDetails = "") {
        int *res_ptr = res.buffer().as<int*>();
        size_t res_size = res.size();

        int *ref_ptr = ref.buffer().as<int*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size) << assertDetails;

        for (size_t i = 0; i < ref_size; i++) {
            ASSERT_EQ(res_ptr[i], ref_ptr[i]) << assertDetails;
        }
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            bucketize_test_params p = ::testing::WithParamInterface<bucketize_test_params>::GetParam();
            std::string model = getModel(p);

                        InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "Bucketize") {
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

            InferenceEngine::Blob::Ptr boundaries = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.boundaries_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.boundaries_shape) });
            boundaries->allocate();
            auto *boundaries_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(boundaries.get());
            std::copy(p.boundaries_value.begin(), p.boundaries_value.end(), (float *)boundaries_ptr->data());
            input_blob_map["BoundariesValues"] = boundaries;

            // prepare output blob map
            InferenceEngine::OutputsDataMap out = network.getOutputsInfo();
            InferenceEngine::BlobMap output_blob_map;
            for (auto iter = out.begin(); iter != out.end(); iter++) {
                std::pair<std::string, InferenceEngine::DataPtr> item = *iter;
                InferenceEngine::Blob::Ptr output_blob_ptr = InferenceEngine::make_shared_blob<int>(item.second->getTensorDesc());
                output_blob_ptr->allocate();
                output_blob_map[item.first] = output_blob_ptr;
            }

            // prepare blobs with reference data
            InferenceEngine::Blob::Ptr output_blob_ref = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                 p.output_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_shape) });
            output_blob_ref->allocate();
            auto *output_blob_ref_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(output_blob_ref.get());
            std::copy(p.output_value_ref.begin(), p.output_value_ref.end(), (int *)output_blob_ref_ptr->data());

            // infer
            graph.Infer(input_blob_map, output_blob_map);

            // check the result
            auto iter = out.begin();
            compare_int(*output_blob_map[iter->first], *output_blob_ref, 0);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtBucketizeTests, TestsBucketize) {}

// model 1 that contains one Bucketize layer
std::string bucketize_model1 = R"V0G0N(
<net Name="Bucketize_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputValues" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="BoundariesValues" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    _BOUNDARIES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="BucketizeLayer" id="2" type="Bucketize" precision="I32">
			<data with_right_bound=_RIGHT_/>
            <input>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
                <port id="1">
                    _BOUNDARIES_SHAPE_
                </port>
            </input>
            <output>
                <port id="2">
                    _OUTPUT_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

// case 1 - the right attribute equal to False
InferenceEngine::SizeVector bucketize_input_shape_case1 = { 10 };
std::vector<float>          bucketize_input_value_case1 = { 8.f, 1.f, 2.f, 1.f, 8.f, 5.f, 1.f, 5.f, 0.f, 20.f };
std::string                 bucketize_right_case1 = "\"false\"";
bool                        bucketize_with_second_input_case1 = true;
InferenceEngine::SizeVector bucketize_boundaries_shape_case1 = { 4 };
std::vector<float>          bucketize_boundaries_value_case1 = { 1.f, 4.f, 10.f, 20.f};
InferenceEngine::SizeVector bucketize_output_shape_case1 = { 10 };
std::vector<int>            bucketize_output_value_ref_case1 = { 2, 1, 1, 1, 2, 2, 1, 2, 0, 4 };

// case 2 - the right attribute equal to True
InferenceEngine::SizeVector bucketize_input_shape_case2 = { 10 };
std::vector<float>          bucketize_input_value_case2 = { 8.f, 1.f, 2.f, 1.f, 8.f, 5.f, 1.f, 5.f, 0.f, 20.f };
std::string                 bucketize_right_case2 = "\"true\"";
bool                        bucketize_with_second_input_case2 = true;
InferenceEngine::SizeVector bucketize_boundaries_shape_case2 = { 4 };
std::vector<float>          bucketize_boundaries_value_case2 = { 1.f, 4.f, 10.f, 20.f };
InferenceEngine::SizeVector bucketize_output_shape_case2 = { 10 };
std::vector<int>            bucketize_output_value_ref_case2 = { 2, 0, 1, 0, 2, 2, 0, 2, 0, 3 };

INSTANTIATE_TEST_CASE_P(
    TestsBucketize, MKLDNNCPUExtBucketizeTests,
    ::testing::Values(
        bucketize_test_params {
            bucketize_model1, "I32", bucketize_right_case1,
            bucketize_input_shape_case1, bucketize_input_value_case1,
            bucketize_with_second_input_case1, bucketize_boundaries_shape_case1, bucketize_boundaries_value_case1,
            bucketize_output_shape_case1, bucketize_output_value_ref_case1,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        bucketize_test_params{
            bucketize_model1, "I32", bucketize_right_case2,
            bucketize_input_shape_case2, bucketize_input_value_case2,
            bucketize_with_second_input_case2, bucketize_boundaries_shape_case2, bucketize_boundaries_value_case2,
            bucketize_output_shape_case2, bucketize_output_value_ref_case2,
            1, MKLDNNPlugin::impl_desc_type::unknown
        }
));
