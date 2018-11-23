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

struct crop_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    std::vector<int> axis;
    std::vector<int> offsets;
    std::vector<int> dims;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};



template <typename data_t>
void ref_crop(InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, crop_test_params prm) {
    data_t *dst_ptr = dst.data();

    std::vector<int> offsets(4);
    for (size_t i = 0; i < prm.offsets.size(); i++) {
        offsets[prm.axis[i]] = prm.offsets[i];
    }
    int OFFSET_N = offsets.at(0);
    int OFFSET_C = offsets.at(1);
    int OFFSET_H = offsets.at(2);
    int OFFSET_W = offsets.at(3);

    const int ON = dst.dims().at(3);
    const int OC = dst.dims().at(2);
    const int OH = dst.dims().at(1);
    const int OW = dst.dims().at(0);

    const int _IN = src.dims().at(0);
    const int IC = src.dims().at(1);
    const int IH = src.dims().at(2);
    const int IW = src.dims().at(3);

    auto dst_off = [=](int n, int c, int h, int w) {
        return (n * OW * OH * OC + c * OW * OH + h * OW + w);
    };
    auto src_off = [=](int n, int c, int h, int w) {
        return (n * IW * IH * IC + c * IW * IH + h * IW + w);
    };

    ASSERT_GE(_IN - OFFSET_N, ON);
    ASSERT_GE(IC - OFFSET_C, OC);
    ASSERT_GE(IH - OFFSET_H, OH);
    ASSERT_GE(IW - OFFSET_W, OW);

    data_t* src_ptr = src.data();
    for (int n = 0; n < ON; ++n) {
        for (int c = 0; c < OC; ++c) {
            for (int h = 0; h < OH; ++h) {
                for (int w = 0; w < OW; ++w) {
                    dst_ptr[dst_off(n, c, h, w)] = src_ptr[src_off(n + OFFSET_N, c + OFFSET_C,
                                                                   h + OFFSET_H, w + OFFSET_W)];
                }
            }
        }
    }
}

class MKLDNNGraphCropTests: public TestsCommon,
                                     public WithParamInterface<crop_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Crop_Only" version="2" precision="FP32" batch="1">
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
        <layer name="crop" id="1" type="Crop" precision="FP32">
            <data axis="_AXC_" offset="_OFC_" dim="_DIMC_" />
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
                    <dim>_DN_</dim>
                    <dim>_DC_</dim>
                    <dim>_DH_</dim>
                    <dim>_DW_</dim>
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
    std::string getModel(crop_test_params p) {
        std::string model = model_t;

        std::string axis, offset, dim;
        std::vector<size_t> outDims = {p.in.n, p.in.c, p.in.h, p.in.w};
        for (size_t i = 0; i < p.offsets.size(); i++) {
            if (!axis.empty())
                axis += ",";
            axis += std::to_string(p.axis[i]);
            if (!offset.empty())
                offset += ",";
            offset += std::to_string(p.offsets[i]);
            if (!dim.empty())
                dim += ",";
            dim += std::to_string(p.dims[i]);
            outDims[p.axis[i]] = p.dims[i];
        }

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_DN_", outDims[0]);
        REPLACE_WITH_NUM(model, "_DC_", outDims[1]);
        REPLACE_WITH_NUM(model, "_DH_", outDims[2]);
        REPLACE_WITH_NUM(model, "_DW_", outDims[3]);
        REPLACE_WITH_STR(model, "_AXC_", axis);
        REPLACE_WITH_STR(model, "_OFC_", offset);
        REPLACE_WITH_STR(model, "_DIMC_", dim);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            crop_test_params p = ::testing::WithParamInterface<crop_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Crop) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW,  dims_src);
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

            ref_crop(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphCropTests, TestCrop) {}


INSTANTIATE_TEST_CASE_P(
        TestCrop, MKLDNNGraphCropTests,
        ::testing::Values(
                crop_test_params{{1, 5, 32, 32}, {1, 2, 3}, {2, 5, 4}, {2, 23, 23}, 1, MKLDNNPlugin::impl_desc_type::unknown, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(1, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }}},
                crop_test_params{{3, 8, 32, 32}, {0, 1, 2, 3}, {1, 0, 20, 20}, {2, 8, 5, 5}, 2, MKLDNNPlugin::impl_desc_type::unknown, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(1, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }} },
                crop_test_params{{1, 5, 32, 32}, {3}, {10}, {20}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                crop_test_params{{1, 5, 32, 20}, {2, 3}, {30, 10}, {2, 10}, 1, MKLDNNPlugin::impl_desc_type::unknown }));

class MKLDNNGraphDynBatchCropTests: public MKLDNNGraphCropTests {
protected:

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            crop_test_params p = ::testing::WithParamInterface<crop_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            network.setBatchSize(MB);

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(network);

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

            auto checkCrop = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Crop;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkCrop);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkCrop);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchCropTests, TestsDynBatchCrop) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchCrop, MKLDNNGraphDynBatchCropTests,
        ::testing::Values(
                crop_test_params{{1, 5, 32, 32}, {1, 2, 3}, {2, 5, 4}, {2, 23, 23}, 1, MKLDNNPlugin::impl_desc_type::unknown, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                            ASSERT_EQ(1, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }}},
                crop_test_params{{1, 5, 32, 32}, {3}, {10}, {20}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                crop_test_params{{1, 5, 32, 20}, {2, 3}, {30, 10}, {2, 10}, 1, MKLDNNPlugin::impl_desc_type::unknown }));
