// Copyright (C) 2018 Intel Corporation
//
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


struct softmax_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    int axis;

    size_t num_prim_desc;

    int selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void check_softmax_fwd(const InferenceEngine::TBlob<data_t> &src, softmax_test_params prm)
{
    const data_t *src_data = src.readOnly();

    size_t W = prm.in.w;
    size_t H = prm.in.h;
    size_t C = prm.in.c;
    size_t MB = prm.in.n;

    auto off = [=](int n, int c, int h, int w)
    {
        return (n * W * H * C + c * W * H + h * W + w);
    };

    auto check_norm = [=](double res) {
        if(res < 0.999f || res > 1.001) {
            ASSERT_TRUE(res > 0.99f && res < 1.01);
        }
    };

    if(prm.axis == 0) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    double result = 0.0f;

                    for (int n = 0; n < MB; ++n) {
                        result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                    }
                    check_norm(result);
                }
            }
        }
    }
    else if(prm.axis == 1) {
        for (int n = 0; n < MB; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    double result = 0.0f;

                    for (int c = 0; c < C; ++c) {
                        result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                    }

                    check_norm(result);
                }
            }
        }
    }
    else if(prm.axis == 2) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int w = 0; w < W; ++w) {
                    double result = 0.0f;

                    for (int h = 0; h < H; ++h) {
                        result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                    }

                    check_norm(result);
                }
            }
        }
    }
    else if(prm.axis == 3) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    double result = 0.0f;

                    for (int w = 0; w < W; ++w) {
                        result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                    }

                    check_norm(result);
                }
            }
        }
    }
}

class MKLDNNGraphSoftMaxTests: public TestsCommon,
                                     public WithParamInterface<softmax_test_params> {
    std::string model_t = R"V0G0N(
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
        <layer name="norm" id="1" type="Softmax" precision="FP32">
            <data PrimitivesPriority="_IMPLS_" axis="_AX_"/>
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
    std::string getModel(softmax_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
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
            softmax_test_params p = ::testing::WithParamInterface<softmax_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::SoftMax) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
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

            check_softmax_fwd(*output, p);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphSoftMaxTests, TestsSoftMax) {}


INSTANTIATE_TEST_CASE_P(
        TestsSoftMax, MKLDNNGraphSoftMaxTests,
        ::testing::Values(
                softmax_test_params{{1, 3, 228, 228}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 3, 228, 228}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 100, 6, 1}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 100, 6, 1}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::ref},
                softmax_test_params{{8, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::ref},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 100, 81, 1}, 2, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 100, 81, 1}, 2, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 1, 1, 1}, 3, 1, MKLDNNPlugin::impl_desc_type::ref},
//                softmax_test_params{{1, 1, 1, 33}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 1, 1, 33}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 1, 10, 81}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 1, 10, 81}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
                ));

class MKLDNNGraphDynBatchSoftMaxTests: public MKLDNNGraphSoftMaxTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            softmax_test_params p = ::testing::WithParamInterface<softmax_test_params>::GetParam();
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

            auto checkSoftmax = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::SoftMax;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkSoftmax);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkSoftmax);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchSoftMaxTests, TestsDynBatchSoftMax) {}


INSTANTIATE_TEST_CASE_P(
        TestsDynBatchSoftMax, MKLDNNGraphDynBatchSoftMaxTests,
        ::testing::Values(
                softmax_test_params{{1, 3, 228, 228}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 3, 228, 228}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 100, 6, 1}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 100, 6, 1}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::ref},
                softmax_test_params{{8, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::ref},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 100, 81, 1}, 2, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 100, 81, 1}, 2, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 1, 1, 1}, 3, 1, MKLDNNPlugin::impl_desc_type::ref},
//                softmax_test_params{{1, 1, 1, 33}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 1, 1, 33}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 1, 10, 81}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 1, 10, 81}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
                ));
