// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"
#include "mock_mkldnn_primitive.hpp"
#include "test_graph.hpp"
#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct activation_test_params {
    mkldnn::algorithm alg;
    float alpha;
    float beta;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename T, typename A> inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(s * alpha);
}

template <typename T, typename A> T elu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(alpha * (::expf(s) - 1));
}

template <typename T>
T logistic_fwd(T s) {
    T v = ::expf(s);
    return v / (v + 1);
}

template <typename T, typename A>
T bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? (T)(alpha) : s;
}

template <typename data_t>
void ref_activation(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, activation_test_params prm) {
    size_t IW = src.dims()[3];
    size_t IH = src.dims()[2];
    size_t IC = src.dims()[1];
    size_t MB = src.dims()[0];

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for(int mb = 0; mb < MB; mb++) {
        for(int c = 0; c < IC; c++) {
            for(int h = 0; h < IH; h++) {
                for(int w = 0; w < IW; w++) {
                    int idx = mb * IC * IH * IW
                              + c * IH * IW
                              + h * IW + w;

                    switch (prm.alg) {
                        case eltwise_relu:         dst_data[idx] = relu_fwd(src_data[idx], prm.alpha);         break;
                        case eltwise_elu:          dst_data[idx] = elu_fwd(src_data[idx], prm.alpha);          break;
                        case eltwise_logistic:     dst_data[idx] = logistic_fwd(src_data[idx]);                break;
                        case eltwise_bounded_relu: dst_data[idx] = bounded_relu_fwd(src_data[idx], prm.alpha); break;
                        default: assert(!"unknown alg_kind");
                    }
                }
            }
        }
    }
}

class MKLDNNGraphActivationTests: public TestsCommon,
                                     public WithParamInterface<activation_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Activation" version="2" precision="FP32" batch="1">
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
        <layer name="activation" id="1" type="_LT_" precision="FP32">
            <data _P1_NAME_="_P1_VAL_" _P2_NAME_="_P2_VAL_" PrimitivesPriority="_IMPLS_"/>
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

protected:
    virtual void TearDown() {
    }

    std::string getModel(activation_test_params p) {
        std::string model = model_t;

        switch (p.alg) {
            case eltwise_relu:         REPLACE_WITH_STR(model, "_LT_", "ReLU"); break;
            case eltwise_elu:          REPLACE_WITH_STR(model, "_LT_", "ELU"); break;
            case eltwise_logistic:     REPLACE_WITH_STR(model, "_LT_", "Sigmoid"); break;
            case eltwise_bounded_relu: REPLACE_WITH_STR(model, "_LT_", "ReLU6"); break;
            default: assert(!"unknown alg_kind");
        }

        if (p.alg == eltwise_relu)
            REPLACE_WITH_STR(model, "_P1_NAME_", "negative_slope");
        else if (p.alg == eltwise_bounded_relu)
            REPLACE_WITH_STR(model, "_P1_NAME_", "n");
        else
            REPLACE_WITH_STR(model, "_P1_NAME_", "alpha");
        REPLACE_WITH_NUM(model, "_P1_VAL_", p.alpha);

        REPLACE_WITH_STR(model, "_P2_NAME_", "beta");
        REPLACE_WITH_NUM(model, "_P2_VAL_", p.beta);

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

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
            activation_test_params p = ::testing::WithParamInterface<activation_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Activation) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
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

            ref_activation(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphActivationTests, TestsActivation) {}

INSTANTIATE_TEST_CASE_P(
        TestsActivation, MKLDNNGraphActivationTests,
        ::testing::Values(
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
        ));

class MKLDNNGraphDynBatchActivationTests: public MKLDNNGraphActivationTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            activation_test_params p = ::testing::WithParamInterface<activation_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
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

            auto checkActivation = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Activation;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkActivation);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkActivation);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchActivationTests, TestsDynBatchActivation) {}


INSTANTIATE_TEST_CASE_P(
        TestsDynBatchActivation, MKLDNNGraphDynBatchActivationTests,
        ::testing::Values(
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {2, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::jit},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_relu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 0.5f, 0.5f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_elu, 1.0f, 1.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_logistic, 0.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 6.0f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {1, 32, 128, 256}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                activation_test_params{eltwise_bounded_relu, 0.1f, 0.0f, {4, 3, 228, 228}, 3, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
        ));
