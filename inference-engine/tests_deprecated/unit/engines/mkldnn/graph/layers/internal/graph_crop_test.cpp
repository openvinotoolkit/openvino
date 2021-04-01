// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct crop_test_params {
    InferenceEngine::SizeVector in;
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

    std::vector<int> offsets(4, 0);
    for (size_t i = 0; i < prm.offsets.size(); i++) {
        offsets[prm.axis[i]] = prm.offsets[i];
    }
    int OFFSET_N = offsets.at(0);
    int OFFSET_C = offsets.at(1);
    int OFFSET_H = offsets.at(2);
    int OFFSET_W = offsets.at(3);

    auto dst_dims = dst.getTensorDesc().getDims();
    int dst_ndims = static_cast<int>(dst_dims.size());
    const int ON = (dst_ndims  > 0) ? dst_dims[0] : 1;
    const int OC = (dst_ndims  > 1) ? dst_dims[1] : 1;
    const int OH = (dst_ndims  > 2) ? dst_dims[2] : 1;
    const int OW = (dst_ndims  > 3) ? dst_dims[3] : 1;

    auto src_dims = src.getTensorDesc().getDims();
    int src_ndims = static_cast<int>(src_dims.size());
    const int _IN = (src_ndims  > 0) ? src_dims[0] : 1;
    const int IC = (src_ndims  > 1) ? src_dims[1] : 1;
    const int IH = (src_ndims  > 2) ? src_dims[2] : 1;
    const int IW = (src_ndims  > 3) ? src_dims[3] : 1;

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
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="crop" id="1" type="Crop" precision="FP32">
            <data axis="_AXC_" offset="_OFC_" dim="_DIMC_" />
            <input>
                <port id="1">
                    _IN_
                </port>
            </input>
            <output>
                <port id="2">
                    _OUT_
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
        std::string in_shape, out_shape;

        std::string axis, offset, dim;
        InferenceEngine::SizeVector outDims = p.in;
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

        for (size_t i = 0; i < p.in.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);

        for (size_t i = 0; i < outDims.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(outDims[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

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

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

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

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,  p.in, InferenceEngine::TensorDesc::getLayoutByDims(p.in) });
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
                crop_test_params{{1, 5, 32, 20}, {2, 3}, {30, 10}, {2, 10}, 1, MKLDNNPlugin::impl_desc_type::unknown },
                crop_test_params{ { 32, 32 },{ 1 },{ 10 },{ 20 }, 1, MKLDNNPlugin::impl_desc_type::unknown },
                crop_test_params{ { 32, 20 },{ 0, 1 },{ 30, 10 },{ 2, 10 }, 1, MKLDNNPlugin::impl_desc_type::unknown }));

class MKLDNNGraphDynBatchCropTests: public MKLDNNGraphCropTests {
protected:

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            crop_test_params p = ::testing::WithParamInterface<crop_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));
            network.setBatchSize(MB);

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(network);

            InferenceEngine::SizeVector dims_src = p.in;
            dims_src[0] = MB;
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::TensorDesc::getLayoutByDims(dims_src) });
            InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
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
