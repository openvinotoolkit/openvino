// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct normalize_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;
    int across_spatial;
    int channel_shared;
    float eps;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_normalize(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, normalize_test_params prm, const float *weights) {
    int B = static_cast<int>(src.getTensorDesc().getDims()[0]);
    int C = static_cast<int>(src.getTensorDesc().getDims()[1]);
    int H = static_cast<int>(src.getTensorDesc().getDims()[2]);
    int W = static_cast<int>(src.getTensorDesc().getDims()[3]);
            
    float eps = prm.eps;
    
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();
    
    for (int b = 0; b < B; b++) {
        const float *src_data_b = src_data + b * C * H * W;
        float *dst_data_b = dst_data + b * C * H * W;
        if (prm.across_spatial) {
            float sqrt_sum = eps;
            for (int i = 0; i < H * W * C; i++) {
                sqrt_sum += (src_data_b[i] * src_data_b[i]);
            }

            sqrt_sum = std::sqrt(sqrt_sum);

            for (int c = 0; c < C; c++) {
                float s = prm.channel_shared ? weights[0] : weights[c];
                for (int hw = 0; hw < H * W; hw++) {
                    dst_data_b[c * H * W + hw] = (src_data_b[c * H * W + hw] / sqrt_sum) * s;
                }
            }
        } else {
            for(int i = 0; i<H*W; i++) {
                int offset = i;

                float norm = eps;
                for (int c = 0; c < C; c++) {
                    const float *src_data_b_c = src_data_b + c * W * H;
                    norm += src_data_b_c[offset] * src_data_b_c[offset];
                }

                norm = std::sqrt(norm);

                for (int c = 0; c < C; c++) {
                    const float *src_data_b_c = src_data_b + c * W * H;
                    float *dst_data_b_c = dst_data_b + c * W * H;

                    dst_data_b_c[offset] = prm.channel_shared ? (src_data_b_c[offset] / norm * weights[0]) : (src_data_b_c[offset] / norm * weights[c]);
                }
            }
        }
    }
}

class MKLDNNCPUExtNormalizeTests: public TestsCommon, public WithParamInterface<normalize_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Normalize_Net" version="2" precision="FP32" batch="1">
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
        <layer name="normalize" id="1" type="Normalize" precision="FP32">
            <data across_spatial="_AS_" channel_shared="_CS_" eps="_EPS_" />
            <weights offset="0" size="_WS_" />

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

    std::string getModel(normalize_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        
        REPLACE_WITH_NUM(model, "_AS_", p.across_spatial);
        REPLACE_WITH_NUM(model, "_CS_", p.channel_shared);

        REPLACE_WITH_NUM(model, "_WS_", p.in.c*sizeof(float));
        REPLACE_WITH_NUM(model, "_EPS_", p.eps);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            normalize_test_params p = ::testing::WithParamInterface<normalize_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            
            size_t weightSize = p.in.c*sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {weightSize}, InferenceEngine::C });
            weights->allocate();
            float center = 0;
            float ampl = 100;
            float omega = 0.5;
            fill_data_sine( weights->data().as<float*>(), weights->size() / sizeof(float), center, ampl, omega);
			
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            for (auto &node : nodes) {
                if (node->getName() == "normalize") {
                    ASSERT_LE(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            ASSERT_LE(3, nodes.size());

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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
            ref_normalize(*srcPtr, dst_ref, p, weights->readOnly().as<const float*>());
            compare(*output, dst_ref);
           
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtNormalizeTests, TestsNormalize) {}

INSTANTIATE_TEST_CASE_P(
        TestsNormalize, MKLDNNCPUExtNormalizeTests,
        ::testing::Values(
                normalize_test_params{{1, 22, 129, 323}, false, false, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 22, 129, 323}, false, true, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{5, 1, 128, 256}, false, false, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{5, 1, 128, 256}, false, true, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 2, 129, 323}, true, false, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 2, 129, 323}, true, true, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown }, 
                normalize_test_params{{2, 1, 21, 21}, true, false, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.000001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 35, 101, 127}, true, true, 0.001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 35, 101, 127}, true, false, 0.001f, 1, MKLDNNPlugin::impl_desc_type::unknown },
                normalize_test_params{{1, 128, 320, 320}, false, true, 0.001f, 1, MKLDNNPlugin::impl_desc_type::unknown }));
