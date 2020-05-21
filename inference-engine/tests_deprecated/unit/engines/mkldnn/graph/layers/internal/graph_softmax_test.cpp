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


struct softmax_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;

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

    auto dims_size = prm.dims.size();

    int axis = prm.axis;
    if (dims_size == 4 && axis > 1)
        axis++;

    size_t W = prm.dims[dims_size - 1];
    size_t H = prm.dims[dims_size - 2];
    size_t D = dims_size == 5 ? prm.dims[dims_size - 3] : 1u;
    size_t C = prm.dims[1];
    size_t MB = prm.dims[0];

    auto off = [=](int n, int c, int d, int h, int w)
    {
        return (n * W * H * D * C + c * W * H * D + d * W * H + h * W + w);
    };

    auto check_norm = [=](double res) {
        if(res < 0.999f || res > 1.001) {
            ASSERT_TRUE(res > 0.99f && res < 1.01);
        }
    };

    if(axis == 0) {
        for (int c = 0; c < C; ++c) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        double result = 0.0f;

                        for (int n = 0; n < MB; ++n) {
                            result += src_data[off(n, c, d, h, w)];
                        }
                        check_norm(result);
                    }
                }
            }
        }
    }
    else if(axis == 1) {
        for (int n = 0; n < MB; ++n) {
            for (int d = 0; d < D; ++d) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        double result = 0.0f;

                        for (int c = 0; c < C; ++c) {
                            result += src_data[off(n, c, d, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                        }

                        check_norm(result);
                    }
                }
            }
        }
    }
    else if(axis == 2) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        double result = 0.0f;

                        for (int d = 0; d < D; ++d) {
                            result += src_data[off(n, c, d, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                        }

                        check_norm(result);
                    }
                }
            }
        }
    }
    else if(axis == 3) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int d = 0; d < D; ++d) {
                    for (int w = 0; w < W; ++w) {
                        double result = 0.0f;

                        for (int h = 0; h < H; ++h) {
                            result += src_data[off(n, c, d, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                        }

                        check_norm(result);
                    }
                }
            }
        }
    }
    else if(axis == 4) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int d = 0; d < D; ++d) {
                    for (int h = 0; h < H; ++h) {
                        double result = 0.0f;

                        for (int w = 0; w < W; ++w) {
                            result += src_data[off(n, c, d, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
                        }

                        check_norm(result);
                    }
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
                    <dim>_ID_</dim>
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
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
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

        auto dims_size = p.dims.size();
        switch (dims_size) {
            case 3:
                REMOVE_LINE(model, "<dim>_IH_</dim>");
            case 4:
                REMOVE_LINE(model, "<dim>_ID_</dim>");
        }

        REPLACE_WITH_NUM(model, "_IW_", p.dims[dims_size - 1]);
        REPLACE_WITH_NUM(model, "_IC_", p.dims[1]);
        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);
        switch (dims_size) {
            case 5:
                REPLACE_WITH_NUM(model, "_ID_", p.dims[dims_size - 3]);
            case 4:
                REPLACE_WITH_NUM(model, "_IH_", p.dims[dims_size - 2]);
        }

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
            
            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);
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

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, layout});
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
                softmax_test_params{{1, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 1000, 1, 1}, 1, 1, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 19, 128, 128}, 1, 2, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 100, 81, 1}, 2, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 100, 81, 1}, 2, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{1, 1, 1, 1}, 3, 1, MKLDNNPlugin::impl_desc_type::jit},
//                softmax_test_params{{1, 1, 1, 33}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{1, 1, 1, 33}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
//                softmax_test_params{{8, 1, 10, 81}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{8, 1, 10, 81}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{2, 5, 9, 10, 11}, 0, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 2, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 4, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
                ));

class MKLDNNGraphDynBatchSoftMaxTests: public MKLDNNGraphSoftMaxTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            softmax_test_params p = ::testing::WithParamInterface<softmax_test_params>::GetParam();
            std::string model = getModel(p);
            InferenceEngine::SizeVector dims_src = p.dims;
            size_t MB = dims_src[0];
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

            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, layout});
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
                softmax_test_params{{8, 1, 10, 81}, 3, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}},
                softmax_test_params{{2, 5, 9, 10, 11}, 1, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 2, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 3, 2, MKLDNNPlugin::impl_desc_type::jit},
                softmax_test_params{{2, 5, 9, 10, 11}, 4, 1, MKLDNNPlugin::impl_desc_type::ref, {MKLDNNPlugin::impl_desc_type::ref_any}}
                ));
