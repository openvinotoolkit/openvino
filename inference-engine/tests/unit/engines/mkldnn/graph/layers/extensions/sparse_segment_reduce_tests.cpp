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


struct sparse_segment_reduce_test_params {
    std::string model;

    std::string precision;

    std::string reduce_op;

    InferenceEngine::SizeVector input_data_shape;
    std::vector<float> input_data_value;
    InferenceEngine::SizeVector input_indices_shape;
    std::vector<float> input_indices_value;
    InferenceEngine::SizeVector input_segment_ids_shape;
    std::vector<float> input_segment_ids_value;

    InferenceEngine::SizeVector output_shape;

    std::vector<float> output_ref;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtSparseSegmentReduceTests : public TestsCommon, public WithParamInterface<sparse_segment_reduce_test_params> {
    std::string getModel(sparse_segment_reduce_test_params p) {
        std::string model = p.model;

        std::string input_data_shape;
        std::string input_indices_shape;
        std::string input_segment_ids_shape;
        std::string output_shape;

        for (auto& shape : p.input_data_shape) {
            input_data_shape += "<dim>";
            input_data_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_indices_shape) {
            input_indices_shape += "<dim>";
            input_indices_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_segment_ids_shape) {
            input_segment_ids_shape += "<dim>";
            input_segment_ids_shape += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_shape) {
            output_shape += "<dim>";
            output_shape += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_REDUCE_OP_", p.reduce_op);
        REPLACE_WITH_STR(model, "_INPUT_DATA_", input_data_shape);
        REPLACE_WITH_STR(model, "_INPUT_INDICES_", input_indices_shape);
        REPLACE_WITH_STR(model, "_INPUT_SEGMENT_IDS_", input_segment_ids_shape);
        REPLACE_WITH_STR(model, "_OUTPUT_", output_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            sparse_segment_reduce_test_params p = ::testing::WithParamInterface<sparse_segment_reduce_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "SparseSegmentReduce") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                        node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            // prepare input blobs and input blob map
            InferenceEngine::BlobMap input_blob_map;
            InferenceEngine::Blob::Ptr input_data = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_data_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_data_shape) });
            input_data->allocate();
            auto *input_data_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_data.get());
            std::copy(p.input_data_value.begin(), p.input_data_value.end(), (float *)input_data_ptr->data());
            input_blob_map["InputData"] = input_data;
            InferenceEngine::Blob::Ptr input_indices = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_indices_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_indices_shape) });
            input_indices->allocate();
            auto *input_indices_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_indices.get());
            std::copy(p.input_indices_value.begin(), p.input_indices_value.end(), (float *)input_indices_ptr->data());
            input_blob_map["InputIndices"] = input_indices;

            InferenceEngine::Blob::Ptr input_segment_ids = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_segment_ids_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_segment_ids_shape) });
            input_segment_ids->allocate();
            auto *input_segment_ids_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_segment_ids.get());
            std::copy(p.input_segment_ids_value.begin(), p.input_segment_ids_value.end(), (float *)input_segment_ids_ptr->data());
            input_blob_map["InputSegmentIds"] = input_segment_ids;

            // prepare output blob map
            InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap output_blob_map;
            for (auto iter = out.begin(); iter != out.end(); iter++) {
                std::pair<std::string, InferenceEngine::DataPtr> item = *iter;
                InferenceEngine::Blob::Ptr output_blob_ptr = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output_blob_ptr->allocate();
                output_blob_map[item.first] = output_blob_ptr;
            }

            // prepare blob with output reference data
            InferenceEngine::Blob::Ptr output_ref = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                 p.output_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_shape) });
            output_ref->allocate();
            auto *output_ref_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(output_ref.get());
            std::copy(p.output_ref.begin(), p.output_ref.end(), (float *)output_ref_ptr->data());

            // infer
            graph.Infer(input_blob_map, output_blob_map);

            // check the result
            auto iter = out.begin();
            compare(*output_blob_map[iter->first], *output_ref, 0.0f);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtSparseSegmentReduceTests, TestsSparseSegmentReduce) {}

// model that contains one SparseSegmentReduce layer
std::string model = R"V0G0N(
<net Name="SparseSegmentReduce_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputData" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _INPUT_DATA_
                </port>
            </output>
        </layer>
        <layer name="InputIndices" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    _INPUT_INDICES_
                </port>
            </output>
        </layer>
        <layer name="InputSegmentIds" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
                    _INPUT_SEGMENT_IDS_
                </port>
            </output>
        </layer>
        <layer name="SparseSegmentReduceLayer" id="3" type="_REDUCE_OP_" precision="FP32">
            <input>
                <port id="0">
                    _INPUT_DATA_
                </port>
                <port id="1">
                    _INPUT_INDICES_
                </port>
                <port id="2">
                    _INPUT_SEGMENT_IDS_
                </port>
            </input>
            <output>
                <port id="0">
                    _OUTPUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

// case 0 - reduce = "sum", 5 segments, where two segments are empty
std::string                 reduce_op_case0 = "SparseSegmentSum";
InferenceEngine::SizeVector input_data_shape_case0        = { 4, 3 };
std::vector<float>          input_data_value_case0        = { 0.f, 1.f, 2.f,
                                                              3.f, 4.f, 5.f,
                                                              6.f, 7.f, 8.f,
                                                              9.f, 10.f, 11.f };
InferenceEngine::SizeVector input_indices_shape_case0     = { 5 };
std::vector<float>          input_indices_value_case0     = { 3.f, 1.f, 1.f, 0.f, 2.f};
InferenceEngine::SizeVector input_segment_ids_shape_case0 = { 5 };
std::vector<float>          input_segment_ids_value_case0 = { 0.f, 0.f, 2.f, 2.f, 4.f };
InferenceEngine::SizeVector output_shape_case0            = { 5, 3 };
std::vector<float>          output_value_ref_case0        = { 12.f, 14.f, 16.f,
                                                              0.f, 0.f, 0.f,
                                                              3.f, 5.f, 7.f,
                                                              0.f, 0.f, 0.f,
                                                              6.f, 7.f, 8.f };

// case 1 - reduce = "mean", 5 segments, where two segments are empty
std::string                 _reduce_op_case1 = "SparseSegmentMean";
InferenceEngine::SizeVector _input_data_shape_case1 = { 4, 3 };
std::vector<float>          _input_data_value_case1 = { 0.f, 1.f, 2.f,
                                                        3.f, 4.f, 5.f,
                                                        6.f, 7.f, 8.f,
                                                        9.f, 10.f, 11.f };
InferenceEngine::SizeVector _input_indices_shape_case1 = { 5 };
std::vector<float>          _input_indices_value_case1 = { 3.f, 1.f, 1.f, 0.f, 2.f };
InferenceEngine::SizeVector _input_segment_ids_shape_case1 = { 5 };
std::vector<float>          _input_segment_ids_value_case1 = { 0.f, 0.f, 2.f, 2.f, 4.f };
InferenceEngine::SizeVector _output_shape_case1 = { 5, 3 };
std::vector<float>          _output_value_ref_case1 = { 6.f, 7.f, 8.f,
                                                        0.f, 0.f, 0.f,
                                                        1.5f, 2.5f, 3.5f,
                                                        0.f, 0.f, 0.f,
                                                        6.f, 7.f, 8.f };

// case 2 - reduce = "sqrtn", 5 segments, where two segments are empty
std::string                 _reduce_op_case2 = "SparseSegmentSqrtN";
InferenceEngine::SizeVector _input_data_shape_case2 = { 4, 3 };
std::vector<float>          _input_data_value_case2 = { 0.f, 1.f, 2.f,
                                                        3.f, 4.f, 5.f,
                                                        6.f, 7.f, 8.f,
                                                        9.f, 10.f, 11.f };
InferenceEngine::SizeVector _input_indices_shape_case2 = { 6 };
std::vector<float>          _input_indices_value_case2 = { 0.f, 1.f, 2.f, 3.f, 1.f, 0.f};
InferenceEngine::SizeVector _input_segment_ids_shape_case2 = { 6 };
std::vector<float>          _input_segment_ids_value_case2 = { 0.f, 0.f, 0.f, 0.f, 2.f, 4.f };
InferenceEngine::SizeVector _output_shape_case2 = { 6, 3 };
std::vector<float>          _output_value_ref_case2 = { 9.f, 11.f, 13.f,
                                                        0.f, 0.f, 0.f,
                                                        3.f, 4.f, 5.f,
                                                        0.f, 0.f, 0.f,
                                                        0.f, 1.f, 2.f,
                                                        0.f, 0.f, 0.f};

INSTANTIATE_TEST_CASE_P(
    TestsSparseSegmentReduce, MKLDNNCPUExtSparseSegmentReduceTests,
    ::testing::Values(
        // case 0 - reduce with sum operation, 5 segments, where two segments are empty
        sparse_segment_reduce_test_params{
            model, "FP32", reduce_op_case0,
            input_data_shape_case0, input_data_value_case0,
            input_indices_shape_case0, input_indices_value_case0,
            input_segment_ids_shape_case0, input_segment_ids_value_case0,
            output_shape_case0, output_value_ref_case0,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 1 - reduce with mean operation, 5 segments, where two segments are empty
        sparse_segment_reduce_test_params{
            model, "FP32", _reduce_op_case1,
            _input_data_shape_case1, _input_data_value_case1,
            _input_indices_shape_case1, _input_indices_value_case1,
            _input_segment_ids_shape_case1, _input_segment_ids_value_case1,
            _output_shape_case1, _output_value_ref_case1,
            1, MKLDNNPlugin::impl_desc_type::unknown
        },
        // case 2 - reduce with sqrtn operation, 5 segments, where two segments are empty
        sparse_segment_reduce_test_params{
            model, "FP32", _reduce_op_case2,
            _input_data_shape_case2, _input_data_value_case2,
            _input_indices_shape_case2, _input_indices_value_case2,
            _input_segment_ids_shape_case2, _input_segment_ids_value_case2,
            _output_shape_case2, _output_value_ref_case2,
            1, MKLDNNPlugin::impl_desc_type::unknown
        }
));
