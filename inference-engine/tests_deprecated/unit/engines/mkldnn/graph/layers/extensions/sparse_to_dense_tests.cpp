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


struct sparse_to_dense_test_params {
    std::string model;
    std::string precision;

    InferenceEngine::SizeVector input_indices_shape;
    std::vector<int> input_indices;
    InferenceEngine::SizeVector input_dense_shape_shape;
    std::vector<int> input_dense_shape;
    InferenceEngine::SizeVector input_values_shape;
    std::vector<int> input_values;
    int input_default_value;

    InferenceEngine::SizeVector output_shape;
    std::vector<int> output_value_ref;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtSparseToDenseTests : public TestsCommon, public WithParamInterface<sparse_to_dense_test_params> {
    std::string getModel(sparse_to_dense_test_params p) {
        std::string model = p.model;

        std::string input_indices_shape;
        std::string input_dense_shape_shape;
        std::string input_values_shape;
        std::string output_shape;

        for (auto& shape : p.input_indices_shape) {
            input_indices_shape += "<dim>";
            input_indices_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_dense_shape_shape) {
            input_dense_shape_shape += "<dim>";
            input_dense_shape_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.input_values_shape) {
            input_values_shape += "<dim>";
            input_values_shape += std::to_string(shape) + "</dim>\n";
        }
        for (auto& shape : p.output_shape) {
            output_shape += "<dim>";
            output_shape += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_INPUT_INDICES_SHAPE_", input_indices_shape);
        REPLACE_WITH_STR(model, "_INPUT_DENSE_SHAPE_SHAPE_", input_dense_shape_shape);
        REPLACE_WITH_STR(model, "_INPUT_VALUES_SHAPE_", input_values_shape);
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
            sparse_to_dense_test_params p = ::testing::WithParamInterface<sparse_to_dense_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "SparseToDense") {
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
            std::copy(p.input_indices.begin(), p.input_indices.end(), (int *)input_indices_ptr->data());
            input_blob_map["InputIndices"] = input_indices;

            InferenceEngine::Blob::Ptr input_dense_shape = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                p.input_dense_shape_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_dense_shape_shape) });
            input_dense_shape->allocate();
            auto *input_dense_shape_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_dense_shape.get());
            std::copy(p.input_dense_shape.begin(), p.input_dense_shape.end(), (int *)input_dense_shape_ptr->data());
            input_blob_map["InputDenseShape"] = input_dense_shape;

            InferenceEngine::Blob::Ptr input_values = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                p.input_values_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_values_shape) });
            input_values->allocate();
            auto *input_values_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_values.get());
            std::copy(p.input_values.begin(), p.input_values.end(), (int *)input_values_ptr->data());
            input_blob_map["InputValues"] = input_values;

            InferenceEngine::Blob::Ptr input_default_value = InferenceEngine::make_shared_blob<int>({ InferenceEngine::Precision::I32,
                { }, InferenceEngine::TensorDesc::getLayoutByDims({ }) });
            input_default_value->allocate();
            auto *input_default_value_ptr = dynamic_cast<InferenceEngine::TBlob<int>*>(input_default_value.get());
            *((int *)input_default_value_ptr->data()) = p.input_default_value;
            input_blob_map["InputDefaultValue"] = input_default_value;

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

TEST_P(MKLDNNCPUExtSparseToDenseTests, TestsSparseToDense) {}

// model 1 that contains one SparseToDense layer
std::string sp2d_model1 = R"V0G0N(
<net Name="SparseToDense_net" version="2" precision="I32" batch="1">
    <layers>
        <layer name="InputIndices" type="Input" precision="I32" id="0">
            <output>
                <port id="0">
                    _INPUT_INDICES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDenseShape" type="Input" precision="I32" id="1">
            <output>
                <port id="0">
                    _INPUT_DENSE_SHAPE_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputValues" type="Input" precision="I32" id="2">
            <output>
                <port id="0">
                    _INPUT_VALUES_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="InputDefaultValue" type="Input" precision="I32" id="3">
            <output>
                <port id="0"/>
            </output>
        </layer>
		<layer id="4" name="SparseToDenseLayer" type="SparseToDense">
			<input>
				<port id="0">
                    _INPUT_INDICES_SHAPE_
				</port>
				<port id="1">
                    _INPUT_DENSE_SHAPE_SHAPE_
				</port>
				<port id="2">
                    _INPUT_VALUES_SHAPE_
				</port>
				<port id="3"/>
			</input>
			<output>
				<port id="4" precision="I32">
                    _OUTPUT_SHAPE_
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";

// case 1 - it contains of the default value input
InferenceEngine::SizeVector sp2d_input_indices_shape_case1 = { 5, 2 };
std::vector<int>          sp2d_input_indices_case1 = { 0, 1,
                                                       1, 2,
                                                       1, 3,
                                                       3, 0,
                                                       3, 4 };
InferenceEngine::SizeVector sp2d_input_dense_shape_shape_case1 = { 2 };
std::vector<int>          sp2d_input_dense_shape_case1 = { 4, 5};
InferenceEngine::SizeVector sp2d_input_values_shape_case1 = { 5 };
std::vector<int>          sp2d_input_values_case1 = { 8,
                                                      1,
                                                      2,
                                                      1,
                                                      8 };
int                         sp2d_input_default_value_case1 = -1;
InferenceEngine::SizeVector sp2d_output_shape_case1 = { 4, 5};
std::vector<int>          sp2d_output_value_ref_case1 = { -1, 8, -1, -1, -1,
                                                          -1, -1, 1, 2, -1,
                                                          -1, -1, -1, -1, -1,
                                                           1, -1, -1, -1, 8};

INSTANTIATE_TEST_CASE_P(
    TestsSparseToDense, MKLDNNCPUExtSparseToDenseTests,
    ::testing::Values(
        sparse_to_dense_test_params{
            sp2d_model1, "I32",
            sp2d_input_indices_shape_case1, sp2d_input_indices_case1,
            sp2d_input_dense_shape_shape_case1, sp2d_input_dense_shape_case1,
            sp2d_input_values_shape_case1, sp2d_input_values_case1,
            sp2d_input_default_value_case1,
            sp2d_output_shape_case1, sp2d_output_value_ref_case1,
            1, MKLDNNPlugin::impl_desc_type::unknown
        }
));
