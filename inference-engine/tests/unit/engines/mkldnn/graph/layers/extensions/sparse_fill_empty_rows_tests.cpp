// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include "tests_common.hpp"

#include <algorithm>
#include <vector>
#include <array>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct sparse_fill_empty_rows_test_params {
    std::string precision;
    InferenceEngine::SizeVector input_indices_shape;
    std::vector<float> input_indices_value;

    InferenceEngine::SizeVector input_values_shape;

    InferenceEngine::SizeVector input_dense_shape_shape;
    std::vector<float> input_dense_shape_value;

    InferenceEngine::SizeVector input_default_value_shape;
    std::vector<float> input_default_value_value;

    InferenceEngine::SizeVector output_indices_shape;
    InferenceEngine::SizeVector output_values_shape;
    InferenceEngine::SizeVector output_empty_rows_indicator_shape;

    size_t num_prim_desc;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_sparse_fill_empty_rows(InferenceEngine::TBlob<float> &input_indices,
    InferenceEngine::TBlob<float> &input_values,
    InferenceEngine::TBlob<float> &dense_shape,
    InferenceEngine::TBlob<float> &default_value,
    InferenceEngine::TBlob<float> &output_indices,
    InferenceEngine::TBlob<float> &output_values,
    InferenceEngine::TBlob<float> &output_empty_rows_indicator) {
    const float *input_indices_ptr = input_indices.data();
    const float *input_values_ptr = input_values.data();
    const float *dense_shape_ptr = dense_shape.data();
    const float *default_value_ptr = default_value.data();
    float dflt_value = default_value_ptr[0];

    float num_rows = dense_shape_ptr[0];
    float num_cols = dense_shape_ptr[1];

    std::vector<size_t> dims = input_values.getTensorDesc().getDims();
    size_t inMaxNumValues = dims[0];
    std::vector<size_t> out_dims = output_values.getTensorDesc().getDims();
    size_t outMaxNumValues = out_dims[0];

    // compute actual number of values by searching out of range indice that serves as a marker
    size_t in_actual_num_values = 0;
    for (in_actual_num_values = 0; in_actual_num_values < inMaxNumValues; in_actual_num_values++) {
        float indice_x = input_indices_ptr[2 * in_actual_num_values];
        float indice_y = input_indices_ptr[2 * in_actual_num_values + 1];
        if (indice_x < 0 || indice_y < 0 || indice_x >= num_rows || indice_y >= num_cols) break;
    }

    // create auxiliary container for sorting
    std::vector<std::array<float, 3>> indices_values(in_actual_num_values); // <row, column, value>
    for (size_t i = 0; i < in_actual_num_values; i++) {
        float row = input_indices_ptr[2 * i];
        float col = input_indices_ptr[2 * i + 1];
        float value = input_values_ptr[i];
        std::array<float, 3> elem = { row, col, value };
        indices_values[i] = elem;
    }

    // sort values by row
    std::sort(indices_values.begin(), indices_values.end(),
        [](const std::array<float, 3>& first, const std::array<float, 3>& second) {
        return first[0] < second[0];
    });

    // unsplit indices and values
    std::vector<float> indices_with_sorted_rows;
    std::vector<float> values_for_sorted_rows;
    for (auto const & elem : indices_values) {
        indices_with_sorted_rows.push_back(elem[0]);
        indices_with_sorted_rows.push_back(elem[1]);
        values_for_sorted_rows.push_back(elem[2]);
    }

    // compute start indice for each row and a number of values at each row
    std::vector<int> values_at_row(num_rows);
    std::fill(values_at_row.begin(), values_at_row.end(), 0);
    float prev_row_with_value = -1.0;
    unsigned int total_num_values = 0;
    std::vector<std::array<float, 3>>::iterator curr_it, prev_it;
    for (float row_ind = 0.0; row_ind < num_rows; row_ind = row_ind + 1.0) {
        curr_it = std::find_if(indices_values.begin(), indices_values.end(),
            [row_ind](std::array<float, 3> elem) { return elem[0] == row_ind; });
        if (curr_it != indices_values.end()) {
            if (prev_row_with_value != -1.0) {
                unsigned int num_values_at_prev_row = std::distance(prev_it, curr_it);
                values_at_row[(int)prev_row_with_value] = num_values_at_prev_row;
                total_num_values += num_values_at_prev_row;
            }
            prev_row_with_value = row_ind;
            prev_it = curr_it;
        }
        else {
            total_num_values++;
        }
    }
    if (prev_row_with_value != -1.0) {
        unsigned int num_values_at_prev_row = std::distance(prev_it, indices_values.end());
        values_at_row[(int)prev_row_with_value] = num_values_at_prev_row;
        total_num_values += num_values_at_prev_row;
    }

    // create output indices
    float *output_indices_ptr = output_indices.data();
    float *output_values_ptr = output_values.data();
    float *output_empty_rows_indicator_ptr = output_empty_rows_indicator.data();

    // zero output buffers
    std::memset(output_indices_ptr, 0, outMaxNumValues * 2 * sizeof(float));
    std::memset(output_values_ptr, 0, outMaxNumValues * sizeof(float));
    std::memset(output_empty_rows_indicator_ptr, 0, num_rows * sizeof(float));

    unsigned int curr_pos_from_copy = 0;
    unsigned int curr_pos_to_copy = 0;
    for (int row_ind = 0; row_ind < (int)num_rows; row_ind++) {
        unsigned int num_values_at_row = values_at_row[row_ind];
        if (num_values_at_row == 0) {
            output_empty_rows_indicator_ptr[row_ind] = 1.0;
            output_values_ptr[curr_pos_to_copy] = dflt_value;
            output_indices_ptr[curr_pos_to_copy * 2] = (float)row_ind;
            output_indices_ptr[curr_pos_to_copy * 2 + 1] = 0.0;
            curr_pos_to_copy++;
        }
        else {
            output_empty_rows_indicator_ptr[row_ind] = 0.0;
            std::copy(values_for_sorted_rows.begin() + curr_pos_from_copy,
                values_for_sorted_rows.begin() + curr_pos_from_copy + num_values_at_row,
                output_values_ptr + curr_pos_to_copy);
            std::copy(indices_with_sorted_rows.begin() + 2 * curr_pos_from_copy,
                indices_with_sorted_rows.begin() + 2 * curr_pos_from_copy + 2 * num_values_at_row, output_indices_ptr + 2 * curr_pos_to_copy);
            curr_pos_to_copy += num_values_at_row;
            curr_pos_from_copy += num_values_at_row;
        }
    }

    // mark the end of output using (-1, -1) indice
    if (total_num_values < outMaxNumValues) {
        output_indices_ptr[total_num_values * 2] = -1.0;
        output_indices_ptr[total_num_values * 2 + 1] = -1.0;
    }
}

class MKLDNNCPUExtSparseFillEmptyRowsTests : public TestsCommon, public WithParamInterface<sparse_fill_empty_rows_test_params> {
    std::string model_t = R"V0G0N(
<net Name="SparseFillEmptyRows_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputIndices" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _IIN_
                </port>
            </output>
        </layer>
        <layer name="InputValues" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    _IVL_
                </port>
            </output>
        </layer>
        <layer name="InputDenseShape" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
                    _IDS_
                </port>
            </output>
        </layer>
        <layer name="InputDefaultValue" type="Input" precision="FP32" id="3">
            <output>
                <port id="0">
                    _IDV_
                </port>
            </output>
        </layer>
        <layer name="SparseFillEmptyRows" id="4" type="SparseFillEmptyRows" precision="FP32">
            <input>
                <port id="0">
                    _IIN_
                </port>
                <port id="1">
                    _IVL_
                </port>
                <port id="2">
                    _IDS_
                </port>
                <port id="3">
                    _IDV_
                </port>
            </input>
            <output>
                <port id="0">
                    _OIN_
                </port>
                <port id="1">
                    _OVL_
                </port>
                <port id="2">
                    _ERI_
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

    std::string getModel(sparse_fill_empty_rows_test_params p) {
        std::string model = model_t;
        std::string input_indices;
        std::string input_values;
        std::string dense_shape;
        std::string default_value;
        std::string output_indices;
        std::string output_values;
        std::string output_empty_rows_indicator;

        InferenceEngine::SizeVector input_dense_shape_shape = { 2 };

        for (auto& shape : p.input_indices_shape) {
            input_indices += "<dim>";
            input_indices += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.input_values_shape) {
            input_values += "<dim>";
            input_values += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : input_dense_shape_shape) {
            dense_shape += "<dim>";
            dense_shape += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.input_default_value_shape) {
            default_value += "<dim>";
            default_value += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_indices_shape) {
            output_indices += "<dim>";
            output_indices += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_values_shape) {
            output_values += "<dim>";
            output_values += std::to_string(shape) + "</dim>\n";
        }

        for (auto& shape : p.output_empty_rows_indicator_shape) {
            output_empty_rows_indicator += "<dim>";
            output_empty_rows_indicator += std::to_string(shape) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IIN_", input_indices);
        REPLACE_WITH_STR(model, "_IVL_", input_values);
        REPLACE_WITH_STR(model, "_IDS_", dense_shape);
        REPLACE_WITH_STR(model, "_IDV_", default_value);
        REPLACE_WITH_STR(model, "_OIN_", output_indices);
        REPLACE_WITH_STR(model, "_OVL_", output_values);
        REPLACE_WITH_STR(model, "_ERI_", output_empty_rows_indicator);

        return model;
    }

    template <typename data_t>
    static void fill_data_dbgval(data_t *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<data_t>(i & (sizeof(data_t) * 8 - 1));
        }
    }
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            sparse_fill_empty_rows_test_params p = ::testing::WithParamInterface<sparse_fill_empty_rows_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "SparseFillEmptyRows") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                        node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            // 4 inputs + 1 op + 3 outputs
            ASSERT_EQ(8, nodes.size());

            // Input Data
            InferenceEngine::Blob::Ptr input_indices = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_indices_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_indices_shape) });
            input_indices->allocate();
            auto *input_indices_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_indices.get());
            std::copy(p.input_indices_value.begin(), p.input_indices_value.end(), (float *) input_indices_ptr->data());

            InferenceEngine::Blob::Ptr input_values = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_values_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_values_shape) });
            input_values->allocate();
            fill_data(input_values->buffer(), input_values->size());

            auto *input_values_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_values.get());
            InferenceEngine::Blob::Ptr input_dense_shape = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_dense_shape_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_dense_shape_shape) });
            input_dense_shape->allocate();
            auto *input_dense_shape_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_dense_shape.get());
            std::copy(p.input_dense_shape_value.begin(), p.input_dense_shape_value.end(), (float *) input_dense_shape_ptr->data());

            InferenceEngine::Blob::Ptr input_default_value = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                p.input_default_value_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.input_default_value_shape) });
            input_default_value->allocate();
            auto *input_default_value_ptr = dynamic_cast<InferenceEngine::TBlob<float>*>(input_default_value.get());
            std::copy(p.input_default_value_value.begin(), p.input_default_value_value.end(), (float *) input_default_value_ptr->data());

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap output_blobs;
            auto iter = out.begin();

            std::pair<std::string, InferenceEngine::DataPtr> item = *(iter++);
            InferenceEngine::Blob::Ptr output_indices = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output_indices->allocate();
            output_blobs[item.first] = output_indices;
            InferenceEngine::TBlob<float> output_indices_ref(item.second->getTensorDesc());
            output_indices_ref.allocate();

            item = *(iter++);
            InferenceEngine::Blob::Ptr output_values = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output_values->allocate();
            output_blobs[item.first] = output_values;
            InferenceEngine::TBlob<float> output_values_ref(item.second->getTensorDesc());
            output_values_ref.allocate();

            item = *(iter++);
            InferenceEngine::Blob::Ptr output_empty_rows_indicator = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output_empty_rows_indicator->allocate();
            output_blobs[item.first] = output_empty_rows_indicator;
            InferenceEngine::TBlob<float> output_empty_rows_indicator_ref(item.second->getTensorDesc());
            output_empty_rows_indicator_ref.allocate();

            // Compute reference result
            ref_sparse_fill_empty_rows(*input_indices_ptr, *input_values_ptr, *input_dense_shape_ptr, *input_default_value_ptr,
                output_indices_ref, output_values_ref, output_empty_rows_indicator_ref);

            // Compute IE result
            InferenceEngine::BlobMap inputs;
            inputs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputIndices", input_indices));
            inputs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputValues", input_values));
            inputs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputDenseShape", input_dense_shape));
            inputs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputDefaultValue", input_default_value));

            // Check the result
            graph.Infer(inputs, output_blobs);
            compare(*output_indices, output_indices_ref, 0.0f);
            compare(*output_values, output_values_ref, 0.0f);
            compare(*output_empty_rows_indicator, output_empty_rows_indicator_ref, 0.0f);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtSparseFillEmptyRowsTests, TestsSparseFillEmptyRows) {}


// case 1 - empty sparse tensor with marker
InferenceEngine::SizeVector input_indices_shape_case1 = {2, 2};
std::vector<float>          input_indices_value_case1 = {-1.f, -1.f};
InferenceEngine::SizeVector input_values_shape_case1 = {2};
InferenceEngine::SizeVector input_dense_shape_shape_case1 = {2};
std::vector<float>          input_dense_shape_value_case1 = {3.f, 4.f};
InferenceEngine::SizeVector input_default_value_shape_case1 = {1};
std::vector<float>          input_default_value_case1 = {0.f};
InferenceEngine::SizeVector output_indices_shape_case1 = {12, 2};
InferenceEngine::SizeVector output_values_shape_case1 = {12};
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case1 = {3};

// case 2 - in one row all values absent without marker
InferenceEngine::SizeVector input_indices_shape_case2 = {6, 2};
std::vector<float>          input_indices_value_case2 = {1.f, 0.f, 0.f, 0.f, 3.f, 1.f, 1.f, 2.f, 3.f, 4.f, 0.f, 1.f};
InferenceEngine::SizeVector input_values_shape_case2 = {6};
InferenceEngine::SizeVector input_dense_shape_shape_case2 = {2};
std::vector<float>          input_dense_shape_value_case2 = {4.f, 5.f};
InferenceEngine::SizeVector input_default_value_shape_case2 = {1};
std::vector<float>          input_default_value_case2 = {0.f};
InferenceEngine::SizeVector output_indices_shape_case2 = {20, 2};
InferenceEngine::SizeVector output_values_shape_case2 = {20};
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case2 = {4};

// case 3 - in one row all values absent with marker
InferenceEngine::SizeVector input_indices_shape_case3 = { 6, 2 };
std::vector<float>          input_indices_value_case3 = { 1.f, 0.f, 0.f, 0.f, 3.f, 1.f, 1.f, 2.f, 3.f, 4.f, -1.f, -1.f };
InferenceEngine::SizeVector input_values_shape_case3 = { 6 };
InferenceEngine::SizeVector input_dense_shape_shape_case3 = { 2 };
std::vector<float>          input_dense_shape_value_case3 = { 4.f, 5.f };
InferenceEngine::SizeVector input_default_value_shape_case3 = { 1 };
std::vector<float>          input_default_value_case3 = { 0.f };
InferenceEngine::SizeVector output_indices_shape_case3 = { 20, 2 };
InferenceEngine::SizeVector output_values_shape_case3 = { 20 };
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case3 = { 4 };

// case 4 - in all rows at least one value presents without marker
InferenceEngine::SizeVector input_indices_shape_case4 = { 7, 2 };
std::vector<float>          input_indices_value_case4 = { 1.f, 0.f, 0.f, 0.f, 3.f, 1.f, 1.f, 2.f, 3.f, 3.f, 2.f, 1.f, 4.f, 3.f };
InferenceEngine::SizeVector input_values_shape_case4 = { 7 };
InferenceEngine::SizeVector input_dense_shape_shape_case4 = { 2 };
std::vector<float>          input_dense_shape_value_case4 = { 5.f, 4.f };
InferenceEngine::SizeVector input_default_value_shape_case4 = { 1 };
std::vector<float>          input_default_value_case4 = { 0.f };
InferenceEngine::SizeVector output_indices_shape_case4 = { 20, 2 };
InferenceEngine::SizeVector output_values_shape_case4 = { 20 };
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case4 = { 5 };

// case 5 - in all rows at least one value presents with marker
InferenceEngine::SizeVector input_indices_shape_case5 = { 8, 2 };
std::vector<float>          input_indices_value_case5 = { 1.f, 0.f, 0.f, 0.f, 3.f, 1.f, 1.f, 2.f, 3.f, 3.f, 2.f, 1.f, 4.f, 3.f, -1.f, -1.f };
InferenceEngine::SizeVector input_values_shape_case5 = { 8 };
InferenceEngine::SizeVector input_dense_shape_shape_case5 = { 2 };
std::vector<float>          input_dense_shape_value_case5 = { 5.f, 4.f };
InferenceEngine::SizeVector input_default_value_shape_case5 = { 1 };
std::vector<float>          input_default_value_case5 = { 0.f };
InferenceEngine::SizeVector output_indices_shape_case5 = { 20, 2 };
InferenceEngine::SizeVector output_values_shape_case5 = { 20 };
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case5 = { 5 };

// case 6 - big sparse tensor with many missed rows without marker
InferenceEngine::SizeVector input_indices_shape_case6 = { 7, 2 };
std::vector<float>          input_indices_value_case6 = { 1.f, 0.f, 0.f, 0.f, 99.f, 19.f, 12.f, 2.f, 37.f, 13.f, 2.f, 1.f, 45.f, 3.f };
InferenceEngine::SizeVector input_values_shape_case6 = { 7 };
InferenceEngine::SizeVector input_dense_shape_shape_case6 = { 2 };
std::vector<float>          input_dense_shape_value_case6 = { 100.f, 20.f };
InferenceEngine::SizeVector input_default_value_shape_case6 = { 1 };
std::vector<float>          input_default_value_case6 = { 0.f };
InferenceEngine::SizeVector output_indices_shape_case6 = { 2000, 2 };
InferenceEngine::SizeVector output_values_shape_case6 = { 2000 };
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case6 = { 100 };

// case 7 - big sparse tensor with many missed rows with marker
InferenceEngine::SizeVector input_indices_shape_case7 = { 8, 2 };
std::vector<float>          input_indices_value_case7 = { 1.f, 0.f, 0.f, 0.f, 99.f, 19.f, 12.f, 2.f, 37.f, 13.f, 2.f, 1.f, 45.f, 3.f, -1.f, -1.f };
InferenceEngine::SizeVector input_values_shape_case7 = { 8 };
InferenceEngine::SizeVector input_dense_shape_shape_case7 = { 2 };
std::vector<float>          input_dense_shape_value_case7 = { 100.f, 20.f };
InferenceEngine::SizeVector input_default_value_shape_case7 = { 1 };
std::vector<float>          input_default_value_case7 = { 0.f };
InferenceEngine::SizeVector output_indices_shape_case7 = { 2000, 2 };
InferenceEngine::SizeVector output_values_shape_case7 = { 2000 };
InferenceEngine::SizeVector output_empty_rows_indicator_shape_case7 = { 100 };

INSTANTIATE_TEST_CASE_P(
    TestsSparseFillEmptyRows, MKLDNNCPUExtSparseFillEmptyRowsTests,
            ::testing::Values(
                // case 1 - empty sparse tensor without marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case1, input_indices_value_case1, input_values_shape_case1,
                input_dense_shape_shape_case1, input_dense_shape_value_case1, input_default_value_shape_case1, input_default_value_case1,
                output_indices_shape_case1, output_values_shape_case1, output_empty_rows_indicator_shape_case1,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 2 - in one row all values absent without marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case2, input_indices_value_case2, input_values_shape_case2,
                input_dense_shape_shape_case2, input_dense_shape_value_case2, input_default_value_shape_case2, input_default_value_case2,
                output_indices_shape_case2, output_values_shape_case2, output_empty_rows_indicator_shape_case2,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 3 - in one row all values absent with marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case3, input_indices_value_case3, input_values_shape_case3,
                input_dense_shape_shape_case3, input_dense_shape_value_case3, input_default_value_shape_case3, input_default_value_case3,
                output_indices_shape_case3, output_values_shape_case3, output_empty_rows_indicator_shape_case3,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 4 - in all rows at least one value presents without marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case4, input_indices_value_case4, input_values_shape_case4,
                input_dense_shape_shape_case4, input_dense_shape_value_case4, input_default_value_shape_case4, input_default_value_case4,
                output_indices_shape_case4, output_values_shape_case4, output_empty_rows_indicator_shape_case4,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 5 - in all rows at least one value presents with marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case5, input_indices_value_case5, input_values_shape_case5,
                input_dense_shape_shape_case5, input_dense_shape_value_case5, input_default_value_shape_case5, input_default_value_case5,
                output_indices_shape_case5, output_values_shape_case5, output_empty_rows_indicator_shape_case5,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 6 - big sparse tensor with many missed rows without marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case6, input_indices_value_case6, input_values_shape_case6,
                input_dense_shape_shape_case6, input_dense_shape_value_case6, input_default_value_shape_case6, input_default_value_case6,
                output_indices_shape_case6, output_values_shape_case6, output_empty_rows_indicator_shape_case6,
                1, MKLDNNPlugin::impl_desc_type::unknown },

                // case 7 - big sparse tensor with many missed rows with marker
                sparse_fill_empty_rows_test_params{ "FP32",
                input_indices_shape_case7, input_indices_value_case7, input_values_shape_case7,
                input_dense_shape_shape_case7, input_dense_shape_value_case7, input_default_value_shape_case7, input_default_value_case7,
                output_indices_shape_case7, output_values_shape_case7, output_empty_rows_indicator_shape_case7,
                1, MKLDNNPlugin::impl_desc_type::unknown }
                ));
