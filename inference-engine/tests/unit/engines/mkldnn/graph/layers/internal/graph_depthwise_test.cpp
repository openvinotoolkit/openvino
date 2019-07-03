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

struct depthwise_test_params {
    mkldnn::algorithm alg;

    // Formats: NCHW, NCDHW
    vector<size_t> dims;

    bool isBroadcast;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_depthwise(const InferenceEngine::TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                   InferenceEngine::TBlob<data_t> &dst, depthwise_test_params prm) {
    auto dims_size = src.dims().size();

    size_t IW = src.dims()[dims_size - 1];
    size_t IH = src.dims()[dims_size - 2];
    size_t ID = dims_size == 5 ? src.dims()[2] : 1u;
    size_t IC = src.dims()[1];
    size_t MB = src.dims()[0];

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    size_t bias_offset = prm.isBroadcast ? 1 : IC;
    const data_t *bias_data = weights_data + bias_offset;
    data_t *dst_data = dst.data();

    size_t c1 = IH * IW;
    size_t c2 = IC * c1;
    size_t c3 = ID * c2;
    for (int mb = 0; mb < MB; mb++) {
        size_t m1 = mb * c3;
        for (int c = 0; c < IC; c++) {
            size_t m2 = m1 + c * c1;
            for (int d = 0; d < ID; d++) {
                size_t m3 = m2 + d * c2;
                for (int h = 0; h < IH; h++) {
                    size_t m4 = m3 + h * IW;
                    for (int w = 0; w < IW; w++) {
                        int idx = m4 + w;

                        int widx = prm.isBroadcast ? 0 : c;
                        int bidx = prm.isBroadcast ? 0 : c;

                        if (prm.alg == depthwise_scale_shift)
                            dst_data[idx] = src_data[idx] * weights_data[widx] + bias_data[bidx];
                        else if (prm.alg == depthwise_prelu)
                            dst_data[idx] = src_data[idx] > 0 ? src_data[idx] : src_data[idx]*weights_data[widx];
                    }
                }
            }
        }
    }
}

class MKLDNNGraphDepthwiseTests: public TestsCommon,
                                     public WithParamInterface<depthwise_test_params> {
    std::string model_t_4D = R"V0G0N(
<Net Name="Lrn_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="depthwise" id="1" type="_LT_" precision="FP32">
            <data _P_NAME_="_P_VAL_"  PrimitivesPriority="_IMPLS_"/>
            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";


    std::string model_t_5D = R"V0G0N(
<Net Name="Lrn_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="depthwise" id="1" type="_LT_" precision="FP32">
            <data _P_NAME_="_P_VAL_"  PrimitivesPriority="_IMPLS_"/>
            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
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
    std::string getModel(depthwise_test_params p) {
        std::string model;
        auto dims_size = p.dims.size();
        if (dims_size == 4) {
            model = model_t_4D;
        } else if (dims_size == 5) {
            model = model_t_5D;
        }

        REPLACE_WITH_NUM(model, "_IW_", p.dims[dims_size - 1]);
        REPLACE_WITH_NUM(model, "_IC_", p.dims[1]);
        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);

        switch (dims_size) {
            case 5:
                REPLACE_WITH_NUM(model, "_ID_", p.dims[dims_size - 3]);
            case 4:
                REPLACE_WITH_NUM(model, "_IH_", p.dims[dims_size - 2]);
        }

        if (p.alg == depthwise_scale_shift) {
            REPLACE_WITH_STR(model, "_LT_", "ScaleShift");
            REPLACE_WITH_STR(model, "_P_NAME_", "broadcast");
            REPLACE_WITH_NUM(model, "_P_VAL_", p.isBroadcast ? 1 : 0);
        }
        else if (p.alg == depthwise_prelu) {
            REPLACE_WITH_STR(model, "_LT_", "PReLU");
            REPLACE_WITH_STR(model, "_P_NAME_", "channel_shared");
            REPLACE_WITH_NUM(model, "_P_VAL_", p.isBroadcast ? 1 : 0);
        }

        size_t array_size =  p.isBroadcast ? 1 : p.dims[1];
        size_t w_data_size = array_size * sizeof(float);
        size_t b_data_size = array_size * sizeof(float);
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

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            depthwise_test_params p = ::testing::WithParamInterface<depthwise_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t weightSize = 2 * p.dims[1] * sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {weightSize});
            weights->allocate();
            fill_data( weights->data().as<float*>(), weights->size() / sizeof(float));

            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Depthwise) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = p.dims;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.dims.size()) {
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

            ref_depthwise(*srcPtr, weights->readOnly().as<const float*>(), weights->size() / sizeof(float), dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDepthwiseTests, TestsDepthwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsDepthwise, MKLDNNGraphDepthwiseTests,
        ::testing::Values(
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, false,3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, false,3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                // 5D
                // mkl-dnn does not support 5D depthwise on jit yet
//                depthwise_test_params{depthwise_scale_shift, {1, 32, 16, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
//                depthwise_test_params{depthwise_scale_shift, {4, 3, 16, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
//                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
//                depthwise_test_params{depthwise_scale_shift, {4, 4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
//                depthwise_test_params{depthwise_scale_shift, {1, 32, 16, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                depthwise_test_params{depthwise_scale_shift, {4, 3, 16, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {4, 4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
        ));

class MKLDNNGraphDynBatchDepthwiseTests: public MKLDNNGraphDepthwiseTests {
protected:

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            depthwise_test_params p = ::testing::WithParamInterface<depthwise_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {p.dims[1] * 4 * sizeof(float)});
            weights->allocate();
            fill_data( weights->data().as<float*>(), weights->size() / sizeof(float));
            float * data = weights->buffer();
            for (size_t i = 0; i < weights->size() / sizeof(float); i++) {
                if (data[i] < 0) {
                    data[i] *= -1;
                }
            }
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

            InferenceEngine::SizeVector dims_src = p.dims;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            src->allocate();
            fill_data(src->buffer(), src->size());

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

            auto checkDepthwise = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Depthwise;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkDepthwise);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkDepthwise);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchDepthwiseTests, TestsDynBatchDepthwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchDepthwise, MKLDNNGraphDynBatchDepthwiseTests,
        ::testing::Values(
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, false,3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::jit},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_scale_shift, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, false,3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {4, 3, 228, 228}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 1, 1, 1}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 4, 5, 5}, false, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {4, 4, 10, 10}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                depthwise_test_params{depthwise_prelu, {1, 32, 128, 256}, true, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
        ));
