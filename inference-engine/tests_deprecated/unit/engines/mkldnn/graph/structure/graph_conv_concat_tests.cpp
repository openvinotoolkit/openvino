// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include "ir_gen_helper.hpp"
#include <cpp/ie_cnn_net_reader.h>

using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace single_layer_tests;
using namespace InferenceEngine;

struct concat_params {
    size_t axis;
};

struct conv_concat_params {
    // Formats: NCHW, NCDHW
    std::vector<size_t> in;

    CommonTestUtils::conv_common_params conv;
    concat_params concat;

    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;
};

class MKLDNNConvConcatTests: public TestsCommon,
                                    public WithParamInterface<conv_concat_params> {
    std::string layers_t = R"V0G0N(
        <layer id="2" name="convolution_1" precision="FP32" type="convolution">
            <data kernel="_K_" strides="_KS_"
             pads_begin="_PB_" pads_end="_PE_"
             dilations="1,1,1" output="_OC_" group="_GC_" PrimitivesPriority="_IMPLS_"/>
            <input>
                <port id="0">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    __CONV_OUT_DIMS__
                </port>
            </output>
            <blobs>
                <weights offset="0" size="262144"/>
            </blobs>
	</layer>
        <layer id="3" name="concat0" precision="FP32" type="Concat">
            <data axis="__AXIS__"/>
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    __CONV_OUT_DIMS__
                </port>
                <port id="1">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __CONCAT_OUT_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
)V0G0N";

    std::string getModel(conv_concat_params p) {
        std::string model = layers_t;

        std::string s_dims;
        for (auto& dim : p.in) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__INP_DIMS__", s_dims);

        s_dims = "";
        size_t conv_axis_val = p.in[p.concat.axis];
        int k_len = p.conv.kernel.size();
        for (size_t i = 2lu; i < p.in.size(); i++) {
            size_t inx = k_len - i + 1;
            size_t dim = (p.in[i] + 2lu * p.conv.pads_begin[inx] - p.conv.kernel[inx]) / p.conv.stride[inx] + 1lu;
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
            if (i == p.concat.axis) {
                conv_axis_val = dim;
            }
        }
	REPLACE_WITH_STR(model, "__CONV_OUT_DIMS__", s_dims);

        s_dims = "";
        for (size_t i = 0lu; i < p.in.size(); i++) {
            size_t val = p.in[i];
            if (i == p.concat.axis) {
                val += conv_axis_val;
            }
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(val) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__CONCAT_OUT_DIMS__", s_dims);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.conv.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.conv.stride);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.conv.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.conv.pads_end);
        REPLACE_WITH_NUM(model, "_GC_", p.conv.group);
        REPLACE_WITH_NUM(model, "_OC_", p.conv.out_c);
        REPLACE_WITH_NUM(model, "_IN_", p.in[0]);
        REPLACE_WITH_NUM(model, "__AXIS__", p.concat.axis);

        std::string impls;
        for (const auto& preferType : p.preferTypes) {
            if (!impls.empty())
                impls += ",";
            impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
        }
        REPLACE_WITH_STR(model, "_IMPLS_", impls);

        model = IRTemplateGenerator::getIRTemplate("convolution_Concat", p.in, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_concat_params p = ::testing::WithParamInterface<conv_concat_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t blob_size = p.conv.out_c * p.in[1] / p.conv.group;
            for (size_t i = 0; i < p.conv.kernel.size(); i++) {
                blob_size *= p.conv.kernel[i];
            }
            blob_size = (blob_size + p.conv.out_c);
            InferenceEngine::SizeVector dims_weights = { blob_size };

            std::vector<InferenceEngine::Blob::Ptr> blob_to_model;
            InferenceEngine::Blob::Ptr weights = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, dims_weights, InferenceEngine::C });
            weights->allocate();
            fill_data(weights->buffer().as<float*>(), weights->size());
            blob_to_model.push_back(weights);

            InferenceEngine::Blob::Ptr bias = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, {p.conv.out_c}, InferenceEngine::C });
            bias->allocate();
            fill_data(bias->buffer().as<float*>(), bias->size());
            blob_to_model.push_back(bias);

            size_t total_size_in_bytes = 0;
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) total_size_in_bytes += blb->byteSize();

            InferenceEngine::TBlob<uint8_t>::Ptr model_blob =
                    InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, {total_size_in_bytes}, InferenceEngine::C });
            model_blob->allocate();
            uint8_t* model_blob_ptr = model_blob->buffer().as<uint8_t*>();
            for (InferenceEngine::Blob::Ptr blb : blob_to_model) {
                memcpy(model_blob_ptr, blb->buffer().as<uint8_t*>(), blb->byteSize());
                model_blob_ptr += blb->byteSize();
            }
            net_reader.SetWeights(model_blob);

            auto network = net_reader.getNetwork();
            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            InferenceEngine::SizeVector dims_src = p.in;

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(
                    {InferenceEngine::Precision::FP32, dims_src, InferenceEngine::TensorDesc::getLayoutByDims(p.in)});
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            for (auto& layer : network) {
                layer->params["PrimitivesPriority"] = "cpu:ref,cpu:ref_any";
            }
            MKLDNNGraphTestClass graph2;
            graph2.CreateGraph(network);

            InferenceEngine::BlobMap outputBlobs2;

            InferenceEngine::TBlob<float>::Ptr output2;
            output2 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output2->allocate();
            outputBlobs2[item.first] = output2;

            graph.Infer(srcs, outputBlobs2);

            compare(*output, *output2, 0.0005f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNConvConcatTests, TestsConvConcat) {}

INSTANTIATE_TEST_CASE_P(
        TestsConvConcat, MKLDNNConvConcatTests,
        ::testing::Values(
                conv_concat_params{{1, 256, 4, 4},
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}},
                conv_concat_params{{2, 256, 4, 4},
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}},
                conv_concat_params{{1, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}},
                conv_concat_params{{2, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}},
                conv_concat_params{{1, 256, 4, 4},
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                conv_concat_params{{2, 256, 4, 4},
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                conv_concat_params{{1, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}},
                conv_concat_params{{2, 256, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 256, false },
                                     {1}, {MKLDNNPlugin::impl_desc_type::gemm_blas}}
        ));

