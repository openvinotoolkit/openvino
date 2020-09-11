// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <stdio.h>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <algorithm>

using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct topk_test_params {
    SizeVector           in_shape;
    std::vector<float>   input_tensor;
    int                  axis;
    std::vector<size_t>  src_k;
    std::string          sort;
    std::string          mode;
    SizeVector           out_shape;
    std::vector<float>   reference_val;
    std::vector<size_t>  reference_idx;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

static inline int count(std::vector<size_t> dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

static inline int count(std::vector<size_t> dims, size_t start_ind = 0) {
    return count(dims, start_ind, dims.size());
}

static void ref_topk(InferenceEngine::TBlob<float> &src, InferenceEngine::TBlob<float> &dst_data, InferenceEngine::TBlob<int> &dst_indx, topk_test_params p) {
    float *src_data = src.data();
    float* dst_val = dst_data.data();
    int* dst_idx = dst_indx.data();

    int dim, axis_dist;
    int src_k = static_cast<int>(p.src_k[0]);


    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();;
    int axis_ = p.axis;
    if (axis_ < 0)
        axis_ += src_dims.size();

    size_t axis = static_cast<size_t>(axis_);

    if (src_dims.size() < (1 + axis))
        FAIL() << " Incorrect input parameters dimensions and axis number!";

    bool mode_max;
    if (p.mode == "max")
        mode_max = true;
    else
        mode_max = false;

    bool sort_value;
    if (p.sort == "value")
        sort_value = true;
    else
        sort_value = false;

    int j;
    for (j = src_dims.size() - 1; j >= 0; j--) {
        if (src_dims[j] != 1) break;
    }
    if (static_cast<size_t>(j) == axis) {
        dim = count(src_dims, static_cast<size_t>(j));
        axis_dist = 1;
    } else {
        int axis_ = (p.axis < 0) ? p.axis + static_cast<int>(src_dims.size()) : p.axis;
        dim = static_cast<int>(src_dims[axis_]);
        axis_dist = count(src_dims, axis_) / dim;
    }

    int num = count(src_dims) / dim;
    std::vector<std::pair<float, int> > src_vector(src_k);

    for (int i = 0; i < num; ++i) {
        src_vector[0] = std::make_pair(src_data[(i / axis_dist * dim) * axis_dist + i % axis_dist], 0);
        for (j = 1; j < src_k; ++j) {
            src_vector[j] = std::make_pair(src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
            if (mode_max) {
                if (src_vector[j].first > src_vector[j - 1].first)
                    std::sort(src_vector.begin(), src_vector.begin() + j + 1, std::greater<std::pair<float, int> >());
            } else {
                if (src_vector[j].first < src_vector[0].first)
                    std::sort(src_vector.begin(), src_vector.begin() + j + 1, std::less<std::pair<float, int> >());
            }
        }

        for (; j < dim; ++j) {
            float value = src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist];
            if (mode_max) {
                if (value > src_vector[src_k - 1].first) {
                    src_vector[src_k - 1] = std::make_pair(value, j);
                    std::sort(src_vector.begin(), src_vector.end(), std::greater<std::pair<float, int> >());
                }
            } else {
                if (value < src_vector[0].first) {
                    src_vector[0] = std::make_pair(value, j);
                    std::sort(src_vector.begin(), src_vector.end(), std::less<std::pair<float, int> >());
                }
            }
        }

        if (!sort_value)
            std::sort(src_vector.begin(), src_vector.begin() + src_k, [](const pair<int, int> &a, const pair<int, int> &b)
            { return (a.second < b.second); });

        for (int j = 0; j < src_k; ++j) {
            if (axis_dist != 1) {
                // Produces max_val per axis
                dst_val[(i / axis_dist * src_k + j) * axis_dist + i % axis_dist] = src_vector[j].first;
                dst_idx[(i / axis_dist * src_k + j) * axis_dist + i % axis_dist] = src_vector[j].second;
            } else {
                // Produces max_ind and max_val
                dst_val[i * src_k + j] = src_vector[j].first;
                dst_idx[i * src_k + j] = src_vector[j].second;
            }
        }
    }
}


class MKLDNNCPUExtTopKTests : public TestsCommon, public WithParamInterface<topk_test_params> {
    std::string model_t = R"V0G0N(
<net Name="TopK_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="value" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="src_k" type="Input" precision="I32" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="TopK" precision="FP32">
            <data axis="_AXIS_" sort="_SORT_" mode="_MODE_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
                <port id="4" precision="I32">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(topk_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        for (auto& dct : p.in_shape) {
            in_shape += "<dim>";
            in_shape += std::to_string(dct) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_SORT_", p.sort);
        REPLACE_WITH_STR(model, "_MODE_", p.mode);
        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            topk_test_params p = ::testing::WithParamInterface<topk_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core ie;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = ie.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            auto it = out.begin();
            std::pair<std::string, InferenceEngine::DataPtr> item0 = *it;
            std::pair<std::string, InferenceEngine::DataPtr> item1 = *(++it);

            InferenceEngine::TBlob<float>::Ptr output0;
            output0 = InferenceEngine::make_shared_blob<float>(item0.second->getTensorDesc());
            output0->allocate();
            outputBlobs[item0.first] = output0;

            InferenceEngine::TBlob<int32_t>::Ptr output1;
            output1 = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, p.out_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.out_shape) });
            output1->allocate();
            outputBlobs[item1.first] = output1;

            // Input Data
            InferenceEngine::Blob::Ptr src;
            src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape) });
            src->allocate();
            if (p.input_tensor.size())
                memcpy(src->buffer(), &p.input_tensor[0], sizeof(float)*p.input_tensor.size());
            else
                fill_data_dbgval(src->buffer(), src->size());
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("value", src));

            InferenceEngine::Blob::Ptr seq_lengthsIdx;
            InferenceEngine::SizeVector seq_lengths_dim(1, 1);
            seq_lengthsIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
            seq_lengthsIdx->allocate();
            memcpy(static_cast<int32_t*>(seq_lengthsIdx->buffer()), &p.src_k[0], sizeof(int32_t));
            auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(seq_lengthsIdx.get());
            if (seq_lengthsIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("src_k", seq_lengthsIdx));

            // Output Reference
            InferenceEngine::TBlob<float> dst_data_ref(item0.second->getTensorDesc());
            dst_data_ref.allocate();
            InferenceEngine::TBlob<int> dst_indx_ref(item1.second->getTensorDesc());
            dst_indx_ref.allocate();
            ref_topk(*srcPtr, dst_data_ref, dst_indx_ref, p);

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output0, dst_data_ref);
            for (int i = 0; i < dst_indx_ref.size(); i++)
                if (dst_indx_ref.data()[i] != (*output1).data()[i])
                    FAIL() << "The difference between res_idx[i] and reference_idx[i]";

            for (int i = 0; i < p.reference_val.size(); i++) {
                if(p.reference_val.data()[i] != (*output0).data()[i])
                    FAIL() << "The difference between res_val[i] and reference_val[i]";
            }

            for (int i = 0; i < p.reference_idx.size(); i++) {
                if (p.reference_idx.data()[i] != (*output1).data()[i])
                    FAIL() << "The difference between res_idx[i] and reference_idx[i]";
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtTopKTests, TestsTopK) {}

INSTANTIATE_TEST_CASE_P(
    TestsTopK, MKLDNNCPUExtTopKTests,
            ::testing::Values(
// Params: in_shape, input_tensor, axis, src_k, sort, mode, out_shape, reference_val, reference_idx
                topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "max",{ 3, 1 },{ 3,7,11 },{ 3,3,3 } },
                topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "max",{ 1, 4 },{ 8,9,10,11 },{ 2,2,2,2 } },
                topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "min",{ 3, 1 },{ 0,4,8 },{ 0,0,0 } },
                topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "min",{ 1, 4 },{ 0,1,2,3 },{ 0,0,0,0 } },
                topk_test_params{ { 2, 3, 128, 256 },{}, 1,{ 1 }, "value", "max",{ 2, 1, 128, 256 },{},{} },
                topk_test_params{ { 3, 5, 128, 256 },{}, 1,{ 1 }, "index", "max",{ 3, 1, 128, 256 },{},{} },
                topk_test_params{ { 1, 3, 129, 257 },{}, 1,{ 1 }, "value", "max",{ 1, 1, 129, 257 },{},{} },
                topk_test_params{ { 2, 5, 129, 257 },{}, 1,{ 1 }, "index", "max",{ 2, 1, 129, 257 },{},{} },
                topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "max",{ 3, 3 },{ 3,2,1,7,6,5,11,10,9 },{ 3,2,1,3,2,1,3,2,1 } },
                topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "min",{ 3, 3 },{ 0,1,2,4,5,6,8,9,10 },{ 0,1,2,0,1,2,0,1,2 } },
                topk_test_params{ { 1, 20, 128, 128 },{}, 1,{ 3 }, "value", "max",{ 1, 3, 128, 128 },{},{} },
                topk_test_params{ { 1, 20, 128, 128 },{}, 1,{ 3 }, "index", "min",{ 1, 3, 128, 128 },{},{} },
                topk_test_params{ { 1, 20, 128, 128 },{}, 1,{ 18 }, "value", "min",{ 1, 18, 128, 128 },{},{} },
                topk_test_params{ { 1, 20, 129, 129 },{}, 1,{ 3 }, "value", "max",{ 1, 3, 129, 129 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 1 }, "value", "max",{ 1, 2, 2, 1 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{}, 3,{ 1 }, "index", "max",{ 1, 2, 2, 1 },{},{} },
                topk_test_params{ { 1, 2, 4, 2 },{}, 2,{ 3 }, "value", "max",{ 1, 2, 3, 2 },{},{} },
                topk_test_params{ { 1, 2, 4, 2 },{}, 2,{ 3 }, "index", "max",{ 1, 2, 3, 2 },{},{} },
                topk_test_params{ { 1, 2, 4, 2 },{}, 2,{ 3 }, "value", "min",{ 1, 2, 3, 2 },{},{} },
                topk_test_params{ { 1, 2, 4, 2 },{}, 2,{ 3 }, "index", "min",{ 1, 2, 3, 2 },{},{} },
                topk_test_params{ { 1, 2, 2, 4 },{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 },{3,3,3,3,3,3,3,3,3,3,3,3},{0,1,2,0,1,2,0,1,2,0,1,2} },
                topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
                topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
                topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
                topk_test_params{ { 1, 20, 32, 32 },{}, 1,{ 18 }, "index", "max",{ 1, 18, 32, 32 },{},{} },
                topk_test_params{ { 1, 20, 129, 129 },{}, 1,{ 18 }, "index", "max",{ 1, 18, 129, 129 },{},{} },
                topk_test_params{ { 1, 20, 32, 32 },{}, 1,{ 18 }, "index", "min",{ 1, 18, 32, 32 },{},{} },
                topk_test_params{ { 1, 20, 129, 129 },{}, 1,{ 18 }, "index", "min",{ 1, 18, 129, 129 },{},{} },
                topk_test_params{ { 1, 20, 129, 129 },{}, 1,{ 18 }, "none", "min",{ 1, 18, 129, 129 },{},{} }
            ));


class MKLDNNCPUExtTopK1OutTests : public TestsCommon, public WithParamInterface<topk_test_params> {
    std::string model_t = R"V0G0N(
<net Name="TopK_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="value" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="src_k" type="Input" precision="I32" id="2">
            <output>
                <port id="2"/>
            </output>
        </layer>
        <layer name="output" id="2" type="TopK" precision="_PRECISION_">
            <data axis="_AXIS_" sort="_SORT_" mode="_MODE_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(topk_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        for (auto& dct : p.in_shape) {
            in_shape += "<dim>";
            in_shape += std::to_string(dct) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_SORT_", p.sort);
        REPLACE_WITH_STR(model, "_MODE_", p.mode);
        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        if (p.reference_val.size())
            REPLACE_WITH_STR(model, "_PRECISION_", "FP32");
        else
            REPLACE_WITH_STR(model, "_PRECISION_", "I32");

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            topk_test_params p = ::testing::WithParamInterface<topk_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Input Data
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_shape,
                                                                                        InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape) });
            src->allocate();
            if (p.input_tensor.size())
                memcpy(src->buffer(), &p.input_tensor[0], sizeof(float)*p.input_tensor.size());
            else
                fill_data_dbgval(src->buffer(), src->size());
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("value", src));
            InferenceEngine::Blob::Ptr seq_lengthsIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, {},
                                                                                                     InferenceEngine::TensorDesc::getLayoutByDims({})});
            seq_lengthsIdx->allocate();
            memcpy(static_cast<int32_t*>(seq_lengthsIdx->buffer()), &p.src_k[0], sizeof(int32_t));
            auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(seq_lengthsIdx.get());
            if (seq_lengthsIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("src_k", seq_lengthsIdx));


            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            auto it = out.begin();
            std::pair<std::string, InferenceEngine::DataPtr> item = *it;

            if (p.reference_val.size()) {
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Infer
                graph.Infer(srcs, outputBlobs);
                for (int i = 0; i < p.reference_val.size(); i++) {
                    if (p.reference_val.data()[i] != (*output).data()[i])
                        FAIL() << "The difference between res_val[i] and reference_val[i]";
                }
            } else {
                InferenceEngine::TBlob<int32_t>::Ptr output;
                output = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, p.out_shape,
                                                                      InferenceEngine::TensorDesc::getLayoutByDims(p.out_shape) });
                output->allocate();
                outputBlobs[item.first] = output;

                // Infer
                graph.Infer(srcs, outputBlobs);
                for (int i = 0; i < p.reference_idx.size(); i++) {
                    if (p.reference_idx.data()[i] != (*output).data()[i])
                        FAIL() << "The difference between res_val[i] and reference_idx[i]";
                }
            }
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtTopK1OutTests, TestsTopK) {}

INSTANTIATE_TEST_CASE_P(
    TestsTopK1Out, MKLDNNCPUExtTopK1OutTests,
    ::testing::Values(
        // Params: in_shape, input_tensor, axis, src_k, sort, mode, out_shape, reference_val, reference_idx
        topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "max",{ 3, 1 },{ 3,7,11 },{} },
        topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "max",{ 3, 1 },{},{ 3,3,3 } },
        topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "max",{ 1, 4 },{ 8,9,10,11 },{} },
        topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "max",{ 1, 4 },{},{ 2,2,2,2 } },
        topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "min",{ 3, 1 },{ 0,4,8 },{} },
        topk_test_params{ { 3, 4 },{}, -1,{ 1 }, "value", "min",{ 3, 1 },{},{ 0,0,0 } },
        topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "min",{ 1, 4 },{ 0,1,2,3 },{} },
        topk_test_params{ { 3, 4 },{},  0,{ 1 }, "value", "min",{ 1, 4 },{},{ 0,0,0,0 } },
        topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "max",{ 3, 3 },{ 3,2,1,7,6,5,11,10,9 },{} },
        topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "max",{ 3, 3 },{},{ 3,2,1,3,2,1,3,2,1 } },
        topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "min",{ 3, 3 },{ 0,1,2,4,5,6,8,9,10 },{} },
        topk_test_params{ { 3, 4 },{}, -1,{ 3 }, "value", "min",{ 3, 3 },{},{ 0,1,2,0,1,2,0,1,2 } },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{} },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 },{},{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{} },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 },{},{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{} },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 },{},{ 0,1,2,0,1,2,0,1,2,0,1,2 } },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 },{ 3,3,3,3,3,3,3,3,3,3,3,3 },{} },
        topk_test_params{ { 1, 2, 2, 4 },{ 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 },{},{ 0,1,2,0,1,2,0,1,2,0,1,2 } }
));
