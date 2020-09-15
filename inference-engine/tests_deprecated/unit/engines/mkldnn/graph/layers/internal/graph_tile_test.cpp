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


struct tile_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t axis;
    size_t tiles;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};


template <typename data_t>
void ref_tile(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst_blob, tile_test_params prm) {
    const float* m_src = src.readOnly();
    int m_outer_dim = 1;
    int m_inner_dim = 1;

    for (int i=0; i < prm.axis; i++ )
        m_outer_dim *= src.getTensorDesc().getDims()[i];
    for (int i=prm.axis; i < src.getTensorDesc().getDims().size(); i++ )
        m_inner_dim *= src.getTensorDesc().getDims()[i];

    float* dst = dst_blob.data();

    for (int i = 0; i < m_outer_dim; ++i) {
        for (int t = 0; t < prm.tiles; ++t) {
            memcpy(dst, m_src, m_inner_dim* sizeof(float));
            dst += m_inner_dim;
        }
        m_src += m_inner_dim;
    }
}

class MKLDNNGraphTileTests: public TestsCommon,
                                     public WithParamInterface<tile_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Tile_Only" version="2" precision="FP32" batch="1">
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
        <layer name="tile" id="1" type="Tile" precision="FP32">
            <data axis="_AX_" tiles="_TL_"/>

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
                    <dim>_ON_</dim>
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
    std::string getModel(tile_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_OW_", (p.axis == 3) ? p.in.w*p.tiles : p.in.w);
        REPLACE_WITH_NUM(model, "_OH_", (p.axis == 2) ? p.in.h*p.tiles : p.in.h);
        REPLACE_WITH_NUM(model, "_OC_", (p.axis == 1) ? p.in.c*p.tiles : p.in.c);
        REPLACE_WITH_NUM(model, "_ON_", (p.axis == 0) ? p.in.n*p.tiles : p.in.n);

        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_NUM(model, "_TL_", p.tiles);

        return model;
    }
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            tile_test_params p = ::testing::WithParamInterface<tile_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Tile) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }

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

            ref_tile(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphTileTests, TestsTile) {}


INSTANTIATE_TEST_CASE_P(
        TestsTile, MKLDNNGraphTileTests,
        ::testing::Values(
                tile_test_params{
                        {1, 128, 1, 1}, 3, 24, 1, MKLDNNPlugin::impl_desc_type::unknown, {
                                         [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                             ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                             ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                             ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                             ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                             ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                         }
                                 }}));

class MKLDNNGraphDynBatchTileTests: public MKLDNNGraphTileTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            tile_test_params p = ::testing::WithParamInterface<tile_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in.n;
            if (MB < 2)
                MB = 2;

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

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

            auto checkTile = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Tile;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkTile);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkTile);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchTileTests, TestsDynBatchTile) {}


// TODO: rewrite to ngraph to have reshape functionality
INSTANTIATE_TEST_CASE_P(
        DISABLED_TestsDynBatchTile, MKLDNNGraphDynBatchTileTests,
        ::testing::Values(
                tile_test_params{
                        {1, 128, 1, 1}, 3, 24, 1, MKLDNNPlugin::impl_desc_type::unknown, {
                                [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                                    ASSERT_EQ(MKLDNNPlugin::impl_desc_type::unknown, impl.getImplementationType());
                                    ASSERT_EQ(1, impl.getConfig().inConfs.size());
                                    ASSERT_EQ(1, impl.getConfig().outConfs.size());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                                    ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                                }
                        }}));
