// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct broadcast_test_params {
    std::string                 shape_precision;
    std::string                 precision;
    InferenceEngine::SizeVector in_shape;
    InferenceEngine::SizeVector out_shape;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};


template <typename data_t>
void ref_broadcast(InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst) {
    size_t i;
    const data_t *src_data = src.data();
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();

    if (!src_dims.size())
        src_dims = InferenceEngine::SizeVector(1, 1);
    if (!srcStrides.size())
        srcStrides = InferenceEngine::SizeVector(1, 1);
    data_t* dst_data = dst.data();
    InferenceEngine::SizeVector dst_dims = dst.getTensorDesc().getDims();
    InferenceEngine::SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();

    if (src_dims.size() > dst_dims.size())
        FAIL() << "Output tensor dimension is smaller then input tensor dimension";

    size_t prefix_size = dst_dims.size() - src_dims.size();
    for (i = 0; i < src_dims.size(); i++) {
        if (src_dims[i] != 1 && src_dims[i] != dst_dims[i + prefix_size])
            FAIL() << "In/Output corresponding dimension must have the same value, or Input dimension is equal to 1";
    }

    InferenceEngine::SizeVector src_aligned(dst_dims.size());
    InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
    for (i = 0; i < dst_dims.size(); i++) {
        if (i < prefix_size) {
            src_aligned[i] = 1;
            srcStrides_aligned[i] = srcStrides[0];
        } else {
            src_aligned[i] = src_dims[i - prefix_size];
            srcStrides_aligned[i] = srcStrides[i - prefix_size];
        }
    }

    size_t src_idx, work_amount_dst = dstStrides[0] * dst_dims[0];
    InferenceEngine::SizeVector counters(dst_dims.size(), 0);

    for (size_t iwork = 0; iwork < work_amount_dst; ++iwork) {
        for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
            src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

        dst_data[iwork] = src_data[src_idx];

        for (int j = dst_dims.size() - 1; j >= 0; j--) {
            counters[j] = (counters[j] + 1) % dst_dims[j];
            if (counters[j] != 0) break;
        }
    }
}


class MKLDNNCPUExtBroadcastTests : public TestsCommon, public WithParamInterface<broadcast_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Broadcast_net" version="2" precision="_IIDXP_" batch="1">
    <layers>
        <layer name="input" type="Input" precision="_IIDXP_" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="shape" type="Input" precision="_ISDXP_" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Broadcast" precision="_IIDXP_">
            <data/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
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

    std::string getModel(broadcast_test_params p) {
        std::string model = model_t;
        std::string in_shape = "";
        std::string out_shape;

        REPLACE_WITH_STR(model, "_IIDXP_", p.precision);
        REPLACE_WITH_STR(model, "_ISDXP_", p.shape_precision);
        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.out_shape.size());

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            broadcast_test_params p = ::testing::WithParamInterface<broadcast_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            // Input Data
            InferenceEngine::Blob::Ptr dims;
            InferenceEngine::SizeVector vector_dim(1, p.out_shape.size());
            if (p.shape_precision == "I32") {
                dims = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, vector_dim, InferenceEngine::TensorDesc::getLayoutByDims(vector_dim) });
                dims->allocate();
                for (size_t i = 0; i < p.out_shape.size(); i++) {
                    static_cast<int32_t*>(dims->buffer())[i] = static_cast<int32_t>(p.out_shape[i]);
                }
                auto * dimsPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(dims.get());
                if (dimsPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";
            }  else if (p.shape_precision == "FP32") {
                dims = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, vector_dim, InferenceEngine::TensorDesc::getLayoutByDims(vector_dim) });
                dims->allocate();
                for (size_t i = 0; i < p.out_shape.size(); i++) {
                    static_cast<float*>(dims->buffer())[i] = static_cast<float>(p.out_shape[i]);
                }
                auto * dimsPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(dims.get());
                if (dimsPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
            }

            InferenceEngine::BlobMap srcs;
            InferenceEngine::Blob::Ptr src;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            if (p.precision == "I32") {
                src = InferenceEngine::make_shared_blob<int32_t>({InferenceEngine::Precision::I32, p.in_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape)});
                src->allocate();
                for (size_t i = 0; i < src->size(); i++)
                    static_cast<int32_t*>(src->buffer())[i] = static_cast<int32_t>(i);
                auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(src.get());
                if (srcPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("shape", dims));

                // Output Blob
                InferenceEngine::TBlob<int32_t>::Ptr output;
                output = InferenceEngine::make_shared_blob<int32_t>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<int32_t> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                ref_broadcast(*srcPtr, dst_ref);

                // Infer
                graph.Infer(srcs, outputBlobs);
                for (int i = 0; i < dst_ref.size(); i++) {
                    if (dst_ref.data()[i] != (*output).data()[i])
                        FAIL() << "The difference between res_ptr[i] and ref_ptr[i]";
                }
            } else if (p.precision == "FP32") {
                src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.in_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape)});
                src->allocate();
                fill_data_dbgval(src->buffer(), src->size());
                auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
                if (srcPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("shape", dims));

                // Output Blob
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                ref_broadcast(*srcPtr, dst_ref);

                // Infer
                graph.Infer(srcs, outputBlobs);
                compare(*output, dst_ref);
            }
            else {
                return;
            }
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtBroadcastTests, TestsBroadcast) {}

INSTANTIATE_TEST_CASE_P(
    TestsBroadcast, MKLDNNCPUExtBroadcastTests,
    ::testing::Values(
        // Params: shape_precision, precision, in_shape, out_shape
        broadcast_test_params{ "I32", "I32",{},{ 2, 3, 4 } },
        broadcast_test_params{ "I32", "I32",{ 4, 1, 2 },{ 4, 2, 2 } },
        broadcast_test_params{ "I32", "I32",{ 4, 2, 1 },{ 4, 2, 2 } },
        broadcast_test_params{ "I32", "I32",{ 4, 2 },{ 2, 4, 2 } },
        broadcast_test_params{ "I32", "I32",{ 4, 1, 1 },{ 4, 2, 1 } },
        broadcast_test_params{ "I32", "I32",{ 2, 1, 3, 1 },{ 2, 2, 2, 3, 1 } },
        broadcast_test_params{ "I32","FP32",{},{ 2, 3, 4 } },
        broadcast_test_params{ "I32","FP32",{ 4, 1, 2 },{ 4, 2, 2 } },
        broadcast_test_params{ "I32","FP32",{ 4, 2, 1 },{ 4, 2, 2 } },
        broadcast_test_params{ "I32","FP32",{ 4, 2 },{ 2, 4, 2 } },
        broadcast_test_params{ "I32","FP32",{ 4, 1, 1 },{ 4, 2, 1 } },
        broadcast_test_params{ "I32","FP32", { 2, 1, 3, 1 },{ 2, 2, 2, 3, 1 } },
        broadcast_test_params{"FP32","FP32",{ 2, 1, 3, 1 },{ 2, 2, 2, 3, 1 } }
));
