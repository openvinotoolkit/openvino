// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <ie_core.hpp>

#include <algorithm>
#include <vector>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct sparse_weighted_reduce_test_params {
    std::string model;
    std::string precision;
    std::string reduce_operation;
    bool with_weights;

    InferenceEngine::SizeVector input_indices_shape;
    std::vector<float> input_indices;
    InferenceEngine::SizeVector input_values_shape;
    std::vector<float> input_values;
    InferenceEngine::SizeVector input_dense_shape_shape;
    std::vector<float> input_dense_shape;
    InferenceEngine::SizeVector input_params_table_shape;
    std::vector<float> input_params_table;
    int input_default_value;
    InferenceEngine::SizeVector input_weights_shape;
    std::vector<float> input_weights;

    InferenceEngine::SizeVector output_shape;
    std::vector<float> output_value_ref;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtExperimentalSparseWeightedReduceTests : public TestsCommon, public WithParamInterface<sparse_weighted_reduce_test_params> {
    std::string getModel(sparse_weighted_reduce_test_params p) {
        std::string model = p.model;

        std::string input_indices_shape;
        std::string input_values_shape;
        std::string input_dense_shape_shape;
        std::string input_params_table_shape;
        std::string input_weights_shape;
        std::string output_shape;

        for (auto& shape : p.input_indices_shape) {
            input_indices_shape += "<dim>";
            input_indices_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_values_shape) {
            input_values_shape += "<dim>";
            input_values_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_dense_shape_shape) {
            input_dense_shape_shape += "<dim>";
            input_dense_shape_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_params_table_shape) {
            input_params_table_shape += "<dim>";
            input_params_table_shape += std::to_string(shape) + "</dim>\n";
        }
        if (p.with_weights) {
            for (auto& shape : p.input_weights_shape) {
                input_weights_shape += "<dim>";
                input_weights_shape += std::to_string(shape) + "</dim>\n";
            }
        }

        for (auto& shape : p.output_shape) {
            output_shape += "<dim>";
            output_shape += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_PRECISION_", p.precision);
        REPLACE_WITH_STR(model, "_REDUCE_OPERATION_", p.reduce_operation);

        REPLACE_WITH_STR(model, "_INPUT_INDICES_SHAPE_", input_indices_shape);
        REPLACE_WITH_STR(model, "_INPUT_VALUES_SHAPE_", input_values_shape);
        REPLACE_WITH_STR(model, "_INPUT_DENSE_SHAPE_SHAPE_", input_dense_shape_shape);
        REPLACE_WITH_STR(model, "_INPUT_PARAMS_TABLE_SHAPE_", input_params_table_shape);
        if (p.with_weights) {
            REPLACE_WITH_STR(model, "_INPUT_WEIGHTS_SHAPE_", input_weights_shape);
        }
        REPLACE_WITH_STR(model, "_OUTPUT_SHAPE_", output_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            sparse_weighted_reduce_test_params p = ::testing::WithParamInterface<sparse_weighted_reduce_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "ExperimentalSparseWeightedReduce") {
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
            InferenceEngine::BlobMap input_blob_map;
            InferenceEngine::Blob::Ptr input_indices = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                p.input_indices_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_indices_shape) });
            input_indices->allocate();
            auto *input_indices_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_indices.get());
            std::vector<int> input_indices_int(p.input_indices.begin(), p.input_indices.end());
            std::copy(input_indices_int.begin(), input_indices_int.end(), (int *)input_indices_ptr->data());
            input_blob_map["InputIndices"] = input_indices;

            InferenceEngine::Blob::Ptr input_values = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                p.input_values_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_values_shape) });
            input_values->allocate();
            auto *input_values_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_values.get());
            std::vector<int> input_values_int(p.input_values.begin(), p.input_values.end());
            std::copy(input_values_int.begin(), input_values_int.end(), (int *)input_values_ptr->data());
            input_blob_map["InputValues"] = input_values;

            InferenceEngine::Blob::Ptr input_dense_shape = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                p.input_dense_shape_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_dense_shape_shape) });
            input_dense_shape->allocate();
            auto *input_dense_shape_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_dense_shape.get());
            std::vector<int> input_dense_shape_int(p.input_dense_shape.begin(), p.input_dense_shape.end());
            std::copy(input_dense_shape_int.begin(), input_dense_shape_int.end(), (int *)input_dense_shape_ptr->data());
            input_blob_map["InputDenseShape"] = input_dense_shape;

            InferenceEngine::Blob::Ptr input_params_table = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_params_table_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_params_table_shape) });
            input_params_table->allocate();
            auto *input_params_table_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_params_table.get());
            std::copy(p.input_params_table.begin(), p.input_params_table.end(), (float *)input_params_table_ptr->data());
            input_blob_map["InputParamsTable"] = input_params_table;

            InferenceEngine::Blob::Ptr input_default_value = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                { }, InferenceEngine::TensorDesc::getLayoutByDims({ }) });
            input_default_value->allocate();
            auto *input_default_value_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_default_value.get());
            *((int *)input_default_value_ptr->data()) = p.input_default_value;
            input_blob_map["InputDefaultValue"] = input_default_value;

            if (p.with_weights) {
                InferenceEngine::Blob::Ptr input_weights = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                    p.input_weights_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_weights_shape) });
                input_weights->allocate();
                auto *input_weights_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_weights.get());
                std::copy(p.input_weights.begin(), p.input_weights.end(), (float *)input_weights_ptr->data());
                input_blob_map["InputWeights"] = input_weights;
            }

            // prepare output blob map
            InferenceEngine::OutputsDataMap out = network.getOutputsInfo();
            InferenceEngine::BlobMap output_blob_map;
            for (auto iter = out.begin(); iter != out.end(); iter++) {
                std::pair<std::string, InferenceEngine::DataPtr> item = *iter;
                InferenceEngine::Blob::Ptr output_blob_ptr = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output_blob_ptr->allocate();
                output_blob_map[item.first] = output_blob_ptr;
            }

            // prepare blobs with reference data
            InferenceEngine::Blob::Ptr output_blob_ref = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                 p.output_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.output_shape) });
            output_blob_ref->allocate();
            auto *output_blob_ref_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(output_blob_ref.get());
            std::copy(p.output_value_ref.begin(), p.output_value_ref.end(), (float *)output_blob_ref_ptr->data());

            // infer
            graph.Infer(input_blob_map, output_blob_map);

            // check the result
            auto iter = out.begin();
            compare(*output_blob_map[iter->first], *output_blob_ref, 0.0f);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtExperimentalSparseWeightedReduceTests, TestsExperimentalSparseWeightedReduce) {}

// model 1 that contains one ExperimentalSparseWeightedReduce layer with the weights input
std::string swr_model1 = R"V0G0N(
<net Name="ExperimentalSparseWeightedReduce_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputIndices" type="Input" precision="_PRECISION_" id="0">
            <output>
                <port id="0">
                    _INPUT_INDICES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputValues" type="Input" precision="_PRECISION_" id="1">
            <output>
                <port id="0">
                    _INPUT_VALUES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDenseShape" type="Input" precision="_PRECISION_" id="2">
            <output>
                <port id="0">
                    _INPUT_DENSE_SHAPE_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputParamsTable" type="Input" precision="FP32" id="3">
            <output>
                <port id="0">
                    _INPUT_PARAMS_TABLE_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDefaultValue" type="Input" precision="I32" id="4">
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer name="InputWeights" type="Input" precision="FP32" id="5">
            <output>
                <port id="0">
                    _INPUT_WEIGHTS_SHAPE_
                </port>
            </output>
        </layer>
        <layer id="6" name="ExperimentalSparseWeightedReduceLayer" type="_REDUCE_OPERATION_">
            <input>
                <port id="0">
                    _INPUT_INDICES_SHAPE_
                </port>
                <port id="1">
                    _INPUT_VALUES_SHAPE_
                </port>
                <port id="2">
                    _INPUT_DENSE_SHAPE_SHAPE_
                </port>
                <port id="3">
                    _INPUT_PARAMS_TABLE_SHAPE_
                </port>
                <port id="4"/>
                <port id="5">
                    _INPUT_WEIGHTS_SHAPE_
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    _OUTPUT_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="6" to-port="4"/>
        <edge from-layer="5" from-port="0" to-layer="6" to-port="5"/>
    </edges>
</net>
)V0G0N";

// model 2 that contains one ExperimentalSparseWeightedReduce layer without the weights input
std::string swr_model2 = R"V0G0N(
<net Name="ExperimentalSparseWeightedReduce_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputIndices" type="Input" precision="_PRECISION_" id="0">
            <output>
                <port id="0">
                    _INPUT_INDICES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputValues" type="Input" precision="_PRECISION_" id="1">
            <output>
                <port id="0">
                    _INPUT_VALUES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDenseShape" type="Input" precision="_PRECISION_" id="2">
            <output>
                <port id="0">
                    _INPUT_DENSE_SHAPE_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputParamsTable" type="Input" precision="FP32" id="3">
            <output>
                <port id="0">
                    _INPUT_PARAMS_TABLE_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDefaultValue" type="Input" precision="I32" id="4">
            <output>
                <port id="0"/>
            </output>
        </layer>
        <layer id="6" name="ExperimentalSparseWeightedReduceLayer" type="_REDUCE_OPERATION_">
            <input>
                <port id="0">
                    _INPUT_INDICES_SHAPE_
                </port>
                <port id="1">
                    _INPUT_VALUES_SHAPE_
                </port>
                <port id="2">
                    _INPUT_DENSE_SHAPE_SHAPE_
                </port>
                <port id="3">
                    _INPUT_PARAMS_TABLE_SHAPE_
                </port>
                <port id="4"/>
            </input>
            <output>
                <port id="6" precision="FP32">
                    _OUTPUT_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="6" to-port="4"/>
    </edges>
</net>
)V0G0N";

// case 1 - ExperimentalSparseWeightedSum, I32, the model with weights input
std::string                 swr_precision_case2 = "I32";
std::string                 swr_reduce_operation_case2 = "ExperimentalSparseWeightedSum";
bool                        swr_with_weights_case2 = true;
InferenceEngine::SizeVector swr_input_indices_shape_case2 = { 5, 2 };
std::vector<float>          swr_input_indices_case2 = { 0.0f, 1.0f,
                                                        1.0f, 2.0f,
                                                        1.0f, 3.0f,
                                                        3.0f, 0.0f,
                                                        3.0f, 4.0f };
InferenceEngine::SizeVector swr_input_values_shape_case2 = { 5 };
std::vector<float>          swr_input_values_case2 = { 3.0f,
                                                       1.0f,
                                                       2.0f,
                                                       1.0f,
                                                       4.0f };
InferenceEngine::SizeVector swr_input_dense_shape_shape_case2 = { 2 };
std::vector<float>          swr_input_dense_shape_case2 = { 4.0f, 5.0f };
InferenceEngine::SizeVector swr_input_params_table_shape_case2 = { 5, 3 };
std::vector<float>          swr_input_params_table_case2 = { 1.0f, 2.0f, 3.0f,
                                                             4.0f, 5.0f, 6.0f,
                                                             6.0f, 5.0f, 4.0f,
                                                             3.0f, 2.0f, 1.0f,
                                                             10.0f, 11.0f, 12.0f };
int                         swr_input_default_value_case2 = 0;
InferenceEngine::SizeVector swr_input_weights_shape_case2 = { 5 };
std::vector<float>          swr_input_weights_case2 = { 1.0f,
                                                        2.0f,
                                                        0.5f,
                                                        1.0f,
                                                        3.0f };
InferenceEngine::SizeVector swr_output_shape_case2 = { 4, 3 };
std::vector<float>          swr_output_value_ref_case2 = { 3.0f, 2.0f, 1.0f,
                                                           11.0f, 12.5f, 14.0f,
                                                           1.0f, 2.0f, 3.0f,
                                                           34.0f, 38.0f, 42.0f };


INSTANTIATE_TEST_CASE_P(
    TestsExperimentalSparseWeightedReduce, MKLDNNCPUExtExperimentalSparseWeightedReduceTests,
    ::testing::Values(
        sparse_weighted_reduce_test_params{
            swr_model1, swr_precision_case2, swr_reduce_operation_case2, swr_with_weights_case2,
            swr_input_indices_shape_case2, swr_input_indices_case2,
            swr_input_values_shape_case2, swr_input_values_case2,
            swr_input_dense_shape_shape_case2, swr_input_dense_shape_case2,
            swr_input_params_table_shape_case2, swr_input_params_table_case2,
            swr_input_default_value_case2,
            swr_input_weights_shape_case2, swr_input_weights_case2,
            swr_output_shape_case2, swr_output_value_ref_case2,
            1, MKLDNNPlugin::impl_desc_type::unknown
        }
));
