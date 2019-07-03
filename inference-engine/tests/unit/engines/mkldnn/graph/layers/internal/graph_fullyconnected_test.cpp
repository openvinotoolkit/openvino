// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct fc_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> in_dims;

    size_t out_c;

    size_t num_prim_desc;

    int selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};


template <typename data_t>
void ref_innerproduct(const InferenceEngine::TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                      InferenceEngine::TBlob<data_t> &dst, fc_test_params prm) {
    auto dims_size = src.dims().size();

    size_t IB = src.dims()[0];
    size_t IC = src.dims()[1];
    size_t ID = dims_size == 5 ? src.dims()[dims_size - 3] : 1u;
    size_t IH = src.dims()[dims_size - 2];
    size_t IW = src.dims()[dims_size - 1];

    size_t OC = prm.out_c;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + IW*IH*ID*IC*OC;
    data_t *dst_data = dst.data();

    IE_ASSERT( IW*IH*ID*IC*OC + OC == weightsSize );
    IE_ASSERT( OC == dst.dims()[0] );

    for (size_t n = 0; n < IB; n++) {
        for (size_t oc = 0; oc < OC; oc++) {
            dst_data[n*OC + oc] = bias_data[oc];
            for (size_t ic = 0; ic < IC; ic++) {
                for (size_t kd = 0; kd < ID; kd++) {
                    for (size_t kh = 0; kh < IH; kh++) {
                        for (size_t kw = 0; kw < IW; kw++) {
                            size_t iidx = n * IC * ID * IH * IW
                                        + ic * ID * IH * IW
                                        + kd * IH * IW
                                        + kh * IW
                                        + kw;
                            size_t widx = oc * IC * ID * IH * IW
                                          + ic * ID * IH * IW 
                                          + kd * IH * IW 
                                          + kh * IW 
                                          + kw;

                            dst_data[n*OC + oc] += src_data[iidx] * weights_data[widx];
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNGraphFullyConnectedTests: public TestsCommon,
                                      public WithParamInterface<fc_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="FullyConnected_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="FullyConnected" id="1" type="InnerProduct" precision="FP32">
            <fc out-size="_OC_" PrimitivesPriority="_IMPLS_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";

protected:
    std::string getModel(fc_test_params p) {
        std::string model = model_t;
        std::string s_dims;
        for (auto& dim : p.in_dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_IN_", p.in_dims[0]);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);

        size_t w_data_size = p.out_c * sizeof(float);
        for (int i = 1; i < p.in_dims.size(); i++)
            w_data_size *= p.in_dims[i];
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);
        std::string impls;
        for (const auto& preferType : p.preferTypes) {
            if (!impls.empty())
                impls += ",";
            impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
        }
        REPLACE_WITH_STR(model, "_IMPLS_", impls);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            fc_test_params p = ::testing::WithParamInterface<fc_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t weights_size = p.out_c;
            for (int i = 1; i < p.in_dims.size(); i++) {
                weights_size *= p.in_dims[i];
            }
            weights_size = (weights_size + p.out_c) * sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {weights_size});
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::FullyConnected) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = p.in_dims;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in_dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_innerproduct(*srcPtr, (const float *)weights->buffer(), weights->size() / sizeof(float), dst_ref, p);

            compare(*output, dst_ref, 0.9f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphFullyConnectedTests, TestsFullyConnected) {}


INSTANTIATE_TEST_CASE_P(
        TestsFullyConnected, MKLDNNGraphFullyConnectedTests,
        ::testing::Values(
                fc_test_params{{1, 3, 227, 227}, 96, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 4, 227, 227}, 8, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 4, 227, 227}, 10, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 3, 227, 227}, 96, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                fc_test_params{{1, 4, 227, 227}, 8, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                fc_test_params{{1, 4, 227, 227}, 10, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                //5D
                fc_test_params{{1, 4, 32, 32, 32}, 10, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 3, 32, 32, 32}, 96, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));

class MKLDNNGraphDynBatchFullyConnectedTests: public MKLDNNGraphFullyConnectedTests {
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            fc_test_params p = ::testing::WithParamInterface<fc_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in_dims[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t weights_size = p.out_c;
            for (int i = 1; i < p.in_dims.size(); i++) {
                weights_size *= p.in_dims[i];
            }
            weights_size = (weights_size + p.out_c) * sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {weights_size});
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
            net_reader.SetWeights(weights_ptr);
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src = p.in_dims;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in_dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            auto checkFC = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::FullyConnected;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkFC);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkFC);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchFullyConnectedTests, TestsDynBatchFullyConnected) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchFullyConnected, MKLDNNGraphDynBatchFullyConnectedTests,
        ::testing::Values(
                fc_test_params{{1, 3, 227, 227}, 96, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 4, 227, 227}, 8, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 4, 227, 227}, 10, 6, MKLDNNPlugin::impl_desc_type::gemm },
                fc_test_params{{1, 3, 227, 227}, 96, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                fc_test_params{{1, 4, 227, 227}, 8, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                fc_test_params{{1, 4, 227, 227}, 10, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));
