// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include <cnn_network_impl.hpp>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include "tests_common.hpp"
#include "ie_system_conf.h"

using namespace ::testing;
using namespace MKLDNNPlugin;
using namespace mkldnn;

struct batchnorm4D_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    double epsilon;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_batchnorm4D(const InferenceEngine::TBlob<data_t> &src, const data_t *variance, const data_t *mean,
                     InferenceEngine::TBlob<data_t> &dst, batchnorm4D_test_params prm) {
    size_t MB = src.getTensorDesc().getDims()[0];
    size_t IC = src.getTensorDesc().getDims()[1];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IW = src.getTensorDesc().getDims()[3];

    const double eps = prm.epsilon;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (int c = 0; c < IC; ++c) {
        data_t v_mean = mean[c];
        data_t v_variance = variance[c];
        data_t sqrt_variance = 0;

        sqrt_variance = 1. / sqrt(v_variance + eps);

        for (int n = 0; n < MB; ++n)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    size_t idx = n * IC * IH * IW
                                 + c * IH * IW
                                 + h * IW + w;
                    dst_data[idx] = (src_data[idx] - v_mean) * sqrt_variance;
                }
    }
}

class MKLDNNGraphBatchNormTests: public TestsCommon,
                                     public WithParamInterface<batchnorm4D_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="BatchNorm4D_Only" version="2" precision="FP32" batch="1">
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
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
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
    std::string getModel(batchnorm4D_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_EPSILON_", p.epsilon);

        REPLACE_WITH_NUM(model, "_OW_", p.in.w);
        REPLACE_WITH_NUM(model, "_OH_", p.in.h);
        REPLACE_WITH_NUM(model, "_OC_", p.in.c);

        size_t w_data_size = p.in.c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);

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
            batchnorm4D_test_params p = ::testing::WithParamInterface<batchnorm4D_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {p.in.c * 2 * sizeof(float)}, InferenceEngine::C });
            weights->allocate();
            fill_data(weights->buffer(), weights->size() / sizeof(float));
            float * data = weights->buffer();
            for (size_t i = 0; i < weights->size() / sizeof(float); i++) {
                if (data[i] < 0) {
                    data[i] *= -1;
                }
            }

            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::BatchNormalization) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_TRUE(nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() | p.selectedType);
                }
            }
            ASSERT_GE(5, nodes.size());

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
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

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_batchnorm4D(*srcPtr, (const float*) weights->buffer(), ((const float*) weights->buffer() + p.in.c), dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphBatchNormTests, TestsBatchNorm) {}

const size_t expect_num_impl = InferenceEngine::with_cpu_x86_avx2() ? 5 : 4;

INSTANTIATE_TEST_CASE_P(
        TestsBatchNorm, MKLDNNGraphBatchNormTests,
        ::testing::Values(
                batchnorm4D_test_params{{1, 32, 128, 256}, 1e-6, expect_num_impl, jit},
                batchnorm4D_test_params{{3, 3,  128, 256}, 1e-6, expect_num_impl, jit},
                batchnorm4D_test_params{{1, 32, 128, 256}, 1e-6, expect_num_impl, ref, {ref_any}},
                batchnorm4D_test_params{{3, 3,  128, 256}, 1e-6, expect_num_impl, ref, {ref_any}}));

class MKLDNNGraphDynBatchBatchNormTests: public MKLDNNGraphBatchNormTests {
protected:

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            batchnorm4D_test_params p = ::testing::WithParamInterface<batchnorm4D_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {p.in.c * 4 * sizeof(float)}, InferenceEngine::C });
            weights->allocate();
            fill_data( weights->data().as<float*>(), weights->size() / sizeof(float));
            float * data = weights->buffer();
            for (size_t i = 0; i < weights->size() / sizeof(float); i++) {
                if (data[i] < 0) {
                    data[i] *= -1;
                }
            }
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
            
            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));
            
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(network);

            InferenceEngine::SizeVector dims_src = {MB, p.in.c, p.in.h, p.in.w};
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            auto* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            src->allocate();
            fill_data(src->buffer(), src->size());

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

            auto checkScaleShift = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::BatchNormalization;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkScaleShift);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkScaleShift);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchBatchNormTests, TestsDynBatchBatchNorm) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchBatchNorm, MKLDNNGraphDynBatchBatchNormTests,
        ::testing::Values(
                batchnorm4D_test_params{{1, 32, 128, 256}, 1e-6, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm4D_test_params{{3, 3, 128, 256}, 1e-6, 5, MKLDNNPlugin::impl_desc_type::jit},
                batchnorm4D_test_params{{1, 32, 128, 256}, 1e-6, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                batchnorm4D_test_params{{3, 3, 128, 256}, 1e-6, 5, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));
