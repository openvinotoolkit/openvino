// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/data_utils.hpp"
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"
#include "ir_gen_helper.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include <ie_core.hpp>

#include <nodes/base.hpp>
#include <cpu_isa_traits.hpp>


using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace single_layer_tests;

using namespace Extensions;
using namespace ::Cpu;
using namespace mkldnn::impl;

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
    bool isBlockedFormat;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    Precision prec_in;
    Precision prec_out;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

extern InferenceEngine::IExtensionPtr make_FakeExtensions();

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
        const data_t *src_data_b = src_data + b * C * H * W;
        data_t *dst_data_b = dst_data + b * C * H * W;
        if (prm.across_spatial) {
            float sqrt_sum = 0.f;
            for (int i = 0; i < H * W * C; i++) {
                sqrt_sum += (src_data_b[i] * src_data_b[i]);
            }

            sqrt_sum = std::sqrt(sqrt_sum) + eps;

            for (int c = 0; c < C; c++) {
                float s = prm.channel_shared ? weights[0] : weights[c];
                for (int hw = 0; hw < H * W; hw++) {
                    float dst_value = (src_data_b[c * H * W + hw] / sqrt_sum) * s;
                    if (prm.prec_out == Precision::FP32) {
                        dst_data_b[c * H * W + hw] = dst_value;
                    } else if (prm.prec_out == Precision::U8) {
                        dst_data_b[c * H * W + hw] = (dst_value > 0) ? roundf(dst_value) : 0;
                    } else if (prm.prec_out == Precision::I8) {
                        dst_data_b[c * H * W + hw] = roundf(dst_value);
                    }
                }
            }
        } else {
            for(int i = 0; i<H*W; i++) {
                int offset = i;

                float norm = 0.f;
                for (int c = 0; c < C; c++) {
                    const data_t *src_data_b_c = src_data_b + c * W * H;
                    norm += src_data_b_c[offset] * src_data_b_c[offset];
                }

                norm = std::sqrt(norm) + eps;

                for (int c = 0; c < C; c++) {
                    const data_t *src_data_b_c = src_data_b + c * W * H;
                    data_t *dst_data_b_c = dst_data_b + c * W * H;

                    float dst_value = prm.channel_shared ? (src_data_b_c[offset] / norm * weights[0]) : (src_data_b_c[offset] / norm * weights[c]);
                    if (prm.prec_out == Precision::FP32) {
                        dst_data_b_c[offset] = dst_value;
                    } else if (prm.prec_out == Precision::U8) {
                        dst_data_b_c[offset] = (dst_value > 0) ? roundf(dst_value) : 0;
                    } else if (prm.prec_out == Precision::I8) {
                        dst_data_b_c[offset] = roundf(dst_value);
                    }
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

        <layer name="fakeLayer" id="1" type="_FL_" precision="FP32">
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
        <layer name="normalize" id="2" type="Normalize" precision="FP32">
            <data across_spatial="_AS_" channel_shared="_CS_" eps="_EPS_" />
            <weights offset="0" size="_WS_" />

            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
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
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
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

        if (p.isBlockedFormat)
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerBLK");
        else
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerPLN");

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

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;

            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            auto defaultExtensions = std::make_shared<InferenceEngine::Extensions::Cpu::MKLDNNExtensions>();
            extMgr->AddExtension(defaultExtensions);
            extMgr->AddExtension(make_FakeExtensions());

            size_t weightSize = p.in.c*sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8,
                {weightSize}, InferenceEngine::C });
            weights->allocate();
            float center = 0;
            float ampl = 100;
            float omega = 0.5;
            CommonTestUtils::fill_data_sine( weights->data().as<float*>(), weights->size() / sizeof(float), center, ampl, omega);

            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network, extMgr);

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
                normalize_test_params{{1, 22, 129, 323}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 22, 129, 323}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{5, 1, 128, 256}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{5, 1, 128, 256}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 2, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 2, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 35, 101, 127}, true, true, 0.001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 35, 101, 127}, true, false, 0.001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 128, 320, 320}, false, true, 0.001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 22, 129, 323}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 22, 129, 323}, false, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{5, 1, 128, 256}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{5, 1, 128, 256}, false, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 2, 129, 323}, true, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 2, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{2, 1, 21, 21}, true, true, 0.001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 35, 101, 127}, true, true, 0.001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 35, 101, 127}, true, false, 0.001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                normalize_test_params{{1, 128, 320, 320}, false, true, 0.001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 }
                ));

static std::string precToStr (Precision prec) {
    return prec == Precision::U8 ? "U8" : prec == Precision::I8 ? "I8" : "FP32";
}

template <typename data_t>
static void fill_int_data(data_t *data, int size, bool is_signed) {
    for (int i = 0 ; i < size; i++) {
        data[i] = i * 13 % 21 - 10 * is_signed;
    }
}

class FakeLayerImpl_Normalize: public Cpu::ExtLayerBase,
                     public WithParamInterface<normalize_test_params> {
public:
    explicit FakeLayerImpl_Normalize(const CNNLayer* layer) {
        try {
            is_blocked = layer->GetParamAsBool("is_blocked");
            addConfig(layer);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    bool is_blocked;

    void addConfig(const CNNLayer* layer) {
        LayerConfig config;

        // Fill tensor parameters into config
        auto fill_port = [&] (std::vector<DataConfig>& port, const DataPtr& data) {
            auto div_up = [](const int a, const int b) -> int {
                if (!b)
                    return 0;
                return (a + b - 1) / b;
            };
            if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

            DataConfig dataConfig;
            dataConfig.inPlace = 0;
            dataConfig.constant = false;

            const TensorDesc& data_desc = data->getTensorDesc();
            const SizeVector& data_dims = data_desc.getDims();

            InferenceEngine::Precision precision = data_desc.getPrecision();
            Layout layout;
            if (is_blocked) {
                int blk_size = cpu::mayiuse(cpu::avx512_common) ? 16 : 8;

                std::vector<size_t> blocks = data_dims;
                std::vector<size_t> order(blocks.size());
                for (size_t i = 0; i < order.size(); i++) order[i] = i;

                order.push_back(1);
                blocks[1] = div_up(blocks[1], blk_size);
                blocks.push_back(blk_size);

                dataConfig.desc = TensorDesc(precision, data_dims, {blocks, order});
            } else {
                dataConfig.desc = TensorDesc(precision, data_dims, data_dims.size() == 5 ? NDHWC : NHWC);
            }

            port.push_back(dataConfig);
        };

        fill_port(config.inConfs, layer->insData[0].lock());
        fill_port(config.outConfs, layer->outData[0]);
        config.inConfs[0].desc.setPrecision(config.outConfs[0].desc.getPrecision());
        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        return OK;
    }
};

REG_FACTORY_FOR(Cpu::ImplFactory<FakeLayerImpl_Normalize>, FakeLayer_Normalize);

class MKLDNNCPUExtNormalizeTests_Blocked: public TestsCommon, public WithParamInterface<normalize_test_params> {
    std::string model_t = R"V0G0N(
        <layer name="fakeLayer1" id="1" type="FakeLayer_Normalize">
            <data is_blocked="_IS_BLOCKED_"/>
            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="_PREC_IN_">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="normalize" id="2" type="Normalize">
            <data across_spatial="_AS_" channel_shared="_CS_" eps="_EPS_" />
            <weights offset="0" size="_WS_" />

            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="_PREC_OUT_">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="fakeLayer2" id="3" type="FakeLayer_Normalize">
            <data is_blocked="_IS_BLOCKED_"/>
            <input>
                <port id="5">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="_PREC_OUT_">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
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
        REPLACE_WITH_STR(model, "_PREC_IN_", precToStr(p.prec_in));
        REPLACE_WITH_STR(model, "_PREC_OUT_", precToStr(p.prec_out));
        REPLACE_WITH_NUM(model, "_IS_BLOCKED_", p.isBlockedFormat);

        model = IRTemplateGenerator::getIRTemplate("Normalize_Only", {p.in.n, p.in.c, p.in.h, p.in.w}, "FP32", model, edges_t, 7);

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

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;

            size_t weightSize = p.in.c*sizeof(float);
            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8,
                {weightSize}, InferenceEngine::C });
            weights->allocate();
            float center = 0;
            float ampl = 100;
            float omega = 0.5;
            CommonTestUtils::fill_data_sine( weights->data().as<float*>(), weights->size() / sizeof(float), center, ampl, omega);

            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

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

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};
            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, NCHW});
            src->allocate();
            if (p.prec_in == Precision::U8) {
                fill_int_data(src->buffer().as<float *>(), src->size(), false);
            } else if (p.prec_in == Precision::I8) {
                fill_int_data(src->buffer().as<float *>(), src->size(), true);
            } else {
                fill_data(src->buffer(), src->size());
            }

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

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
            ref_normalize(*srcPtr, dst_ref, p, weights->readOnly().as<const float*>());
            compare(*output, dst_ref);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtNormalizeTests_Blocked, TestsNormalize) {}

INSTANTIATE_TEST_CASE_P(
        TestsNormalize, MKLDNNCPUExtNormalizeTests_Blocked,
        ::testing::Values(
            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
            normalize_test_params{{2, 33, 129, 323}, true, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, false, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.000001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },

            normalize_test_params{{2, 33, 129, 323}, true, true, 0.0001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
            normalize_test_params{{2, 67, 77, 78}, false, false, 0.0001f, true, 3, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 }
        ));