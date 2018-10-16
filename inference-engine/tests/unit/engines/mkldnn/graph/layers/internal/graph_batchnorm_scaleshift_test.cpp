// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <inference_engine/cnn_network_impl.hpp>
#include "mkldnn_plugin/mkldnn_graph.h"
#include "mock_mkldnn_primitive.hpp"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct batchnorm_scaleshift_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    // BatchNorm specific param
    double epsilon;
    // ScaleShift specific param
    int broadcast;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_batchnorm4DWithScale(const InferenceEngine::TBlob<data_t> &src, const data_t *variance, const data_t *mean, const data_t *scaleShift,
                              InferenceEngine::TBlob<data_t> &dst, double eps) {
    size_t MB = src.dims()[0];
    size_t IC = src.dims()[1];
    size_t IH = src.dims()[2];
    size_t IW = src.dims()[3];

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    const data_t *scale_data = scaleShift;
    const data_t *shift_data = scaleShift + IC;
#   pragma omp parallel for schedule(static)
    for (int c = 0; c < IC; ++c) {
        data_t v_mean = mean[c];
        data_t v_variance = variance[c];
        data_t sqrt_variance = 0;
        data_t scale = scale_data[c];
        data_t shift = shift_data[c];

        sqrt_variance = 1. / sqrt(v_variance + eps);

        for (int n = 0; n < MB; ++n)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    size_t idx = n * IC * IH * IW
                                 + c * IH * IW
                                 + h * IW + w;
                    // BatchNorm
                    dst_data[idx] = (src_data[idx] - v_mean) * sqrt_variance;
                    // ScaleShift
                    dst_data[idx] = dst_data[idx] * scale + shift;
                }
    }
}

class MKLDNNGraphBatchNormScaleShiftTests: public TestsCommon,
                                     public WithParamInterface<batchnorm_scaleshift_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="BatchNorm_With_Scale_Fusion" version="2" precision="FP32" batch="1">
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
        <layer name="batchNorm" id="1" type="BatchNormalization" precision="FP32">
            <batch_norm_data epsilon="_EPSILON_" PrimitivesPriority="_IMPLS_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S1_" />

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
         <layer name="scaleshift" id="2" type="ScaleShift" precision="FP32">
            <scale_shift_data broadcast="_BROADCAST_" PrimitivesPriority="_IMPLS_"/>

            <weights offset="_S2_" size="_S1_" />
            <biases offset="_S3_" size="_S1_" />

            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
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
       <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
    </edges>
</Net>
)V0G0N";

protected:
    virtual void TearDown() {
    }

    std::string getModel(batchnorm_scaleshift_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_EPSILON_", p.epsilon);
        REPLACE_WITH_NUM(model, "_BROADCAST_", p.broadcast);

        size_t w_data_size = p.in.c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", 2*w_data_size);
        REPLACE_WITH_NUM(model, "_S3_", 3*w_data_size);

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
            batchnorm_scaleshift_test_params p = ::testing::WithParamInterface<batchnorm_scaleshift_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {p.in.c * 4 * sizeof(float)});
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

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if ((nodes[i]->getType() == MKLDNNPlugin::Depthwise && nodes[i]->getCnnLayer()->type == "ScaleShift")
                    || nodes[i]->getType() == MKLDNNPlugin::BatchNormalization) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_TRUE(nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() | p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
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

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_batchnorm4DWithScale(*srcPtr, (const float*) weights->buffer(), ((const float*) weights->buffer() + p.in.c), (const float*) weights->buffer() + p.in.c*2, dst_ref, p.epsilon);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphBatchNormScaleShiftTests, TestsBatchNormWithScaleShift) {}

INSTANTIATE_TEST_CASE_P(
        TestsBatchNormWithScaleShift, MKLDNNGraphBatchNormScaleShiftTests,
        ::testing::Values(
                batchnorm_scaleshift_test_params{{1, 32, 128, 256}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm_scaleshift_test_params{{4, 3, 227, 227}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm_scaleshift_test_params{{1, 32, 128, 256}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                batchnorm_scaleshift_test_params{{4, 3, 227, 227}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));


class MKLDNNGraphDynBatchBatchNormScaleShiftTests: public MKLDNNGraphBatchNormScaleShiftTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            batchnorm_scaleshift_test_params p = ::testing::WithParamInterface<batchnorm_scaleshift_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {p.in.c * 4 * sizeof(float)});
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

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
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

            auto checkScaleShift = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return (node->getType() == MKLDNNPlugin::Depthwise && node->getCnnLayer()->type == "ScaleShift")
                       || node->getType() == MKLDNNPlugin::BatchNormalization;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkScaleShift);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkScaleShift);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchBatchNormScaleShiftTests, TestsDynBatchBatchNormWithScaleShift) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchBatchNormWithScaleShift, MKLDNNGraphDynBatchBatchNormScaleShiftTests,
        ::testing::Values(
                batchnorm_scaleshift_test_params{{1, 32, 128, 256}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm_scaleshift_test_params{{4, 3, 227, 227}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm_scaleshift_test_params{{1, 32, 128, 256}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                batchnorm_scaleshift_test_params{{4, 3, 227, 227}, 1e-6, 2, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));
