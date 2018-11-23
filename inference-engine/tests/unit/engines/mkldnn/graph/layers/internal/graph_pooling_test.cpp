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

struct pooling_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;
    std::vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_pool(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, pooling_test_params prm)
{
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;

    size_t IW = prm.in.w;
    size_t IH = prm.in.h;

    size_t OW = (IW + 2 * prm.pad_w - prm.krn_w) / prm.str_w + 1;
    size_t OH = (IH + 2 * prm.pad_h - prm.krn_h) / prm.str_h + 1;
    size_t OC = prm.in.c;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    IE_ASSERT( OC == dst.dims()[2]);

    for (size_t c = 0; c < OC; c++) {
        for (size_t oh = 0; oh < OH; oh++) {
            for (size_t ow = 0; ow < OW; ow++) {
                size_t oidx = c * OH * OW
                              + oh * OW + ow;
                data_t out_ref = data_t(0);
                bool is_initialized = false;
                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        int32_t iw = ow * prm.str_w - prm.pad_w + kw;
                        int32_t ih = oh * prm.str_h - prm.pad_h + kh;
                        if (iw < 0 || iw >= IW || ih < 0
                            || ih >= IH)
                            continue;
                        uint32_t iidx = c * IH * IW + ih * IW + iw;

                        data_t d = src_data[iidx];
                        if (!is_initialized) {
                            out_ref = d;
                            is_initialized = true;
                        } else {
                            if (out_ref < d)
                                out_ref = d;
                        }
                    }
                }
                dst_data[oidx] = out_ref;
            }
        }
    }
}

class MKLDNNGraphPoolingTests: public TestsCommon,
                                     public WithParamInterface<pooling_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Pooling_Only" version="2" precision="FP32" batch="1">
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
        <layer name="pool" id="1" type="Pooling" precision="FP32">

            <pooling stride-x="_SW_" stride-y="_SH_"
                     pad-x="_PW_" pad-y="_PH_"
                     kernel-x="_KW_" kernel-y="_KH_"
                     method="MAX" round="Ceil" PrimitivesPriority="_IMPLS_"/>

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
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
    std::string getModel(pooling_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_KW_", p.krn_w);
        REPLACE_WITH_NUM(model, "_KH_", p.krn_h);
        REPLACE_WITH_NUM(model, "_SW_", p.str_w);
        REPLACE_WITH_NUM(model, "_SH_", p.str_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pad_w);
        REPLACE_WITH_NUM(model, "_PH_", p.pad_h);

        REPLACE_WITH_NUM(model, "_OW_", (p.in.w + 2 * p.pad_w - p.krn_w) / p.str_w + 1);
        REPLACE_WITH_NUM(model, "_OH_", (p.in.h + 2 * p.pad_h - p.krn_h) / p.str_h + 1);

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
            pooling_test_params p = ::testing::WithParamInterface<pooling_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Pooling) {
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

            ref_pool(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphPoolingTests, TestsPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestsPooling, MKLDNNGraphPoolingTests,
        ::testing::Values(
                pooling_test_params{{1, 3, 228, 228}, 2, 2, 2, 2, 0, 0, 6, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 2, 0, 0, 4, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 1, 0, 0, 4, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, 2, 2, 2, 2, 0, 0, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 2, 0, 0, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 1, 0, 0, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));


class MKLDNNGraphDynBatchPoolingTests: public MKLDNNGraphPoolingTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            pooling_test_params p = ::testing::WithParamInterface<pooling_test_params>::GetParam();
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

            auto checkPooling = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Pooling;
            };
            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkPooling);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkPooling);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchPoolingTests, TestsDynBatchPooling) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchPooling, MKLDNNGraphDynBatchPoolingTests,
        ::testing::Values(
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 1, 0, 0, 4, MKLDNNPlugin::impl_desc_type::jit},
                pooling_test_params{{1, 3, 228, 228}, 2, 2, 2, 2, 0, 0, 6, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 2, 0, 0, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                pooling_test_params{{1, 3, 228, 228}, 4, 2, 2, 1, 0, 0, 4, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}));
