// Copyright (C) 2018 Intel Corporation
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

struct eltwise_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;

    enum opType {
        Sum = 0, Prod = 1, Max = 2
    };

    opType op;

    std::string scales;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template<typename data_t>
void ref_eltwise(const std::vector<InferenceEngine::TBlob<data_t>> &src, InferenceEngine::TBlob<data_t> &dst, eltwise_test_params prm) {
    std::vector<float> scales;
    if (prm.scales != "") {
        std::istringstream stream(prm.scales);
        std::string str;
        while (getline(stream, str, ',')) {
            float val = std::stof(str);
            scales.push_back(val);
        }
    } else {
        for (int i = 0; i < src.size(); i++) {
            scales.push_back(1.0f);
        }
    }

    data_t *dst_data = dst.data();

    const data_t *src_data = src[0].readOnly();

    for (int i = 0; i < src[0].size(); i++) {
        switch (prm.op) {
            case eltwise_test_params::Sum: {
                dst_data[i] = scales[0]*src_data[i];
            }
                break;
            default: {
                dst_data[i] = src_data[i];
            }
        }
    }

    for (int n = 1; n < src.size(); n++) {
        src_data = src[n].readOnly();

        for (int i = 0; i < src[n].size(); i++) {
            switch (prm.op) {
                case eltwise_test_params::Sum: {
                    dst_data[i] += scales[n]*src_data[i];
                }
                    break;

                case eltwise_test_params::Prod: {
                    dst_data[i] *= src_data[i];
                }
                    break;

                case eltwise_test_params::Max: {
                    dst_data[i] = (std::max)(dst_data[i], src_data[i]);
                }
                    break;
            }
        }
    }
}

class MKLDNNGraphEltwiseTests: public TestsCommon,
                                     public WithParamInterface<eltwise_test_params> {
    std::string model_t = R"V0G0N(
<net name="EltwiseOnly" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="con" id="4" type="Eltwise" precision="FP32">
            <data operation="_OP_" _COEFF_/>
            <input>
                <port id="1">__SRC_DIMS__
                </port>
                <port id="2">__SRC_DIMS__
                </port>
                <port id="3">__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="4">__SRC_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(eltwise_test_params p) {
        std::string model = model_t;
        std::string op;

        if (p.op == 0) {
            op = "sum";
        } else if (p.op == 1) {
            op = "mul";
        } else if (p.op == 2) {
            op = "max";
        }

        std::string src_dims;
        for (auto& dim : p.dims) {
                src_dims += "\n                    <dim>";
                src_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);

        std::string scale;
        if (!p.scales.empty()) {
            scale = std::string("coeff=\"") + p.scales + std::string("\"");
        }
        REPLACE_WITH_STR(model, "_OP_", op);
        REPLACE_WITH_STR(model, "_COEFF_", scale);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Eltwise) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
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

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src1->allocate();

            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());

            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            fill_data(src1->buffer(), src1->size());
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src2->allocate();

            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());

            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());
            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src3->allocate();

            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());

            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src3->buffer(), src3->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

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

            std::vector<InferenceEngine::TBlob<float>> src_vec = {*srcPtr1, *srcPtr2, *srcPtr3};

            ref_eltwise(src_vec, dst_ref, p);

            compare(*output, dst_ref, 0.0005f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphEltwiseTests, TestsEltwise) {}


INSTANTIATE_TEST_CASE_P(
        TestsEltwise, MKLDNNGraphEltwiseTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.0,1.0,1.0", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.5,0.5,-2.0", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Prod, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Max, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 32, 16, 16, 16}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } }
        ));

class MKLDNNGraphDynBatchEltwiseTests: public MKLDNNGraphEltwiseTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims[0];
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

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src1->allocate();

            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());

            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            fill_data(src1->buffer(), src1->size());
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src2->allocate();

            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());

            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());
            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, layout, dims_src);
            src3->allocate();

            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());

            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src3->buffer(), src3->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;


            auto checkDepthwise = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Eltwise;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkDepthwise);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkDepthwise);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchEltwiseTests, TestsDynBatchEltwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchEltwise, MKLDNNGraphDynBatchEltwiseTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.0,1.0,1.0", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.5,0.5,-2.0", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Prod, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3}, eltwise_test_params::opType::Max, "", 3, MKLDNNPlugin::impl_desc_type::ref}));


struct precisions_test_2params {
    struct {
        std::string precision0;
        std::string precision1;
    } in;

    size_t num_nodes;
    size_t num_reorder_nodes;
};

class MKLDNNGraphEltwise2PrecisionsTests : public TestsCommon,
                                     public WithParamInterface<precisions_test_2params> {

    std::string model_t = R"V0G0N(
<net name="default" version="2" batch="1">
    <layers>
        <layer name="second_input" type="Input" precision="_IP1_" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="data" type="Input" precision="_IP0_" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Eltwise" precision="FP32" id="2">
            <elementwise_data operation="sum" coeff=""/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(precisions_test_2params p) {
        std::string model = model_t;

        REPLACE_WITH_STR(model, "_IP0_", p.in.precision0);
        REPLACE_WITH_STR(model, "_IP1_", p.in.precision1);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            precisions_test_2params p = ::testing::WithParamInterface<precisions_test_2params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            ASSERT_EQ(nodes.size(), p.num_nodes);

            size_t actual_reorder_nodes = 0;
            for (size_t i = 0; i < nodes.size(); i++) {
                if(nodes[i].get()->getType() == MKLDNNPlugin::Type::Reorder &&
                    FIND_STR(nodes[i].get()->getName(), "_U8_FP32_"))
                    actual_reorder_nodes ++;
            }
            ASSERT_EQ(actual_reorder_nodes, p.num_reorder_nodes);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphEltwise2PrecisionsTests, TestsEltwise2Precisions) {}

INSTANTIATE_TEST_CASE_P(
        TestsEltwise2Precisions, MKLDNNGraphEltwise2PrecisionsTests,
        ::testing::Values(
            precisions_test_2params{ {"FP32", "FP32"}, 4, 0 },
            precisions_test_2params{ {  "U8", "FP32"}, 5, 1 },
            precisions_test_2params{ {"FP32",   "U8"}, 5, 1 },
            precisions_test_2params{ {  "U8",   "U8"}, 6, 2 }
        ));

