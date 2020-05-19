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
#include "ir_gen_helper.hpp"
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

struct mvn_test_params {
    vector<size_t> dims;

    int across_channels;
    int normalize_variance;
    float eps;

    size_t num_prim_desc;
    bool isBlockedFormat;
    int selectedType;

    Precision prec_in;
    Precision prec_out;

    vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

extern InferenceEngine::IExtensionPtr make_FakeExtensions();

template <typename data_t>
void ref_mvn(const TBlob<data_t> &src, TBlob<data_t> &dst, mvn_test_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();
    size_t dims_size = prm.dims.size();

    size_t N = prm.dims[0];
    size_t C = prm.dims[1];
    size_t D = dims_size > 4 ? prm.dims[dims_size - 3lu] : 1lu;
    size_t H = dims_size > 3 ? prm.dims[dims_size - 2lu] : 1lu;
    size_t W = prm.dims[dims_size - 1lu];

    float eps = prm.eps;

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    float C2inv = 1.f / static_cast<float>(C2);
    float C3inv = 1.f / static_cast<float>(C3);

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        // Calculate mean value
        if (prm.across_channels) {
            float mean = 0.0f;
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean += src_data[ch + w];
                        }
                    }
                }
            }
            mean *= C3inv;
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t index = ch + w;
                            dst_data[index] = src_data[index] - mean;
                        }
                    }
                }
            }
        } else {
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                float mean = 0.0f;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean += src_data[ch + w];
                        }
                    }
                }

                mean *= C2inv;

                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t index = ch + w;
                            dst_data[index] = src_data[index] - mean;
                        }
                    }
                }
            }
        }
    }

    if (prm.normalize_variance) {
        for (size_t b = 0; b < N; b++) {
            size_t cb = b * C3;
            // Calculate variances value
            if (prm.across_channels) {
                float variance = 0.f;
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance += dst_data[ch + w] * dst_data[ch + w];
                            }
                        }
                    }
                }
                variance = 1.f / sqrtf(variance * C3inv + eps);
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] *= variance;
                            }
                        }
                    }
                }
            } else {
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    float variance = 0.0f;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance += dst_data[ch + w] * dst_data[ch + w];
                            }
                        }
                    }
                    variance = 1.f / sqrtf(variance * C2inv + eps);
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] *= variance;
                                if (prm.prec_out == Precision::U8) {
                                    dst_data[ch + w] = (dst_data[ch + w] > 0) ? roundf(dst_data[ch + w]) : 0;
                                } else if (prm.prec_out == Precision::I8) {
                                    dst_data[ch + w] = roundf(dst_data[ch + w]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNCPUExtMVNTests: public TestsCommon, public WithParamInterface<mvn_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="fakeLayer" id="1" type="_FL_" precision="FP32">
            <input>
                <port id="1">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="mvn" id="2" type="MVN" precision="FP32">
            <data across_channels="_AC_" normalize_variance="_NV_" eps="_EPS_"/>
            <input>
                <port id="3">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="4">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
)V0G0N";

    std::string getModel(mvn_test_params p) {
        std::string model = layers_t;
        if (p.isBlockedFormat)
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerBLK");
        else
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerPLN");

        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	    REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AC_", p.across_channels);
        REPLACE_WITH_NUM(model, "_NV_", p.normalize_variance);
        REPLACE_WITH_NUM(model, "_EPS_", p.eps);

        model = IRTemplateGenerator::getIRTemplate("MVN_Only", p.dims, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            mvn_test_params p = ::testing::WithParamInterface<mvn_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            auto defaultExtensions = std::make_shared<InferenceEngine::Extensions::Cpu::MKLDNNExtensions>();
            extMgr->AddExtension(defaultExtensions);
            extMgr->AddExtension(make_FakeExtensions());


            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network, extMgr);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "mvn") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            SizeVector dims_src = p.dims;

            Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = NCHW;
                    break;
                case 5:
                    layout = NCDHW;
                    break;
            }

            Blob::Ptr src = make_shared_blob<float>({ Precision::FP32, dims_src, layout });
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_mvn(*srcPtr, dst_ref, p);
            compare(*output, dst_ref, 0.0001f);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtMVNTests, TestsMVN) {}

INSTANTIATE_TEST_CASE_P(
        TestsMVN, MKLDNNCPUExtMVNTests,
        ::testing::Values(
        /*0*/   mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*9*/   mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*14*/  mvn_test_params{{2,640, 15, 15}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },

                // 5D
        /*16*/  mvn_test_params{{2, 64, 24, 32, 40}, 0, 0, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 1, 0.00001f, 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 0, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 1, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 0, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 1, 0.00001f, 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 0, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*23*/  mvn_test_params{{2, 64, 24, 32, 40}, 1, 1, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{1, 64, 32, 32, 32}, 0, 1, 0.001f, 3, true, MKLDNNPlugin::impl_desc_type::unknown }
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

class FakeLayerImpl_MVN: public Cpu::ExtLayerBase,
                     public WithParamInterface<mvn_test_params> {
public:
    explicit FakeLayerImpl_MVN(const CNNLayer* layer) {
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

REG_FACTORY_FOR(Cpu::ImplFactory<FakeLayerImpl_MVN>, FakeLayer_MVN);

class MKLDNNCPUExtMVNTests_Blocked: public TestsCommon, public WithParamInterface<mvn_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="fakeLayer1" id="1" type="FakeLayer_MVN">
            <data is_blocked="_IS_BLOCKED_"/>
            <input>
                <port id="1">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2" precision="_PREC_IN_">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="mvn" id="2" type="MVN">
            <data across_channels="_AC_" normalize_variance="_NV_" eps="_EPS_"/>
            <input>
                <port id="3">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="4" precision="_PREC_OUT_">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="fakeLayer2" id="3" type="FakeLayer_MVN">
            <data is_blocked="_IS_BLOCKED_"/>
            <input>
                <port id="5">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="6" precision="_PREC_OUT_">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
)V0G0N";

    std::string getModel(mvn_test_params p) {
        std::string model = layers_t;

        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AC_", p.across_channels);
        REPLACE_WITH_NUM(model, "_NV_", p.normalize_variance);
        REPLACE_WITH_NUM(model, "_EPS_", p.eps);
        REPLACE_WITH_STR(model, "_PREC_IN_", precToStr(p.prec_in));
        REPLACE_WITH_STR(model, "_PREC_OUT_", precToStr(p.prec_out));
        REPLACE_WITH_NUM(model, "_IS_BLOCKED_", p.isBlockedFormat);

        model = IRTemplateGenerator::getIRTemplate("MVN_Only", p.dims, "FP32", model, edges_t, 7);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            mvn_test_params p = ::testing::WithParamInterface<mvn_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "mvn") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            SizeVector dims_src = p.dims;

            Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = NCHW;
                    break;
                case 5:
                    layout = NCDHW;
                    break;
            }

            Blob::Ptr src = make_shared_blob<float>({ Precision::FP32, dims_src, layout });
            src->allocate();
            if (p.prec_in == Precision::U8) {
                fill_int_data(src->buffer().as<float *>(), src->size(), false);
            } else if (p.prec_in == Precision::I8) {
                fill_int_data(src->buffer().as<float *>(), src->size(), true);
            } else {
                fill_data(src->buffer(), src->size());
            }

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_mvn(*srcPtr, dst_ref, p);
            compare(*output, dst_ref, 0.0001f);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtMVNTests_Blocked, TestsMVN) {}

INSTANTIATE_TEST_CASE_P(
        TestsMVN, MKLDNNCPUExtMVNTests_Blocked,
        ::testing::Values(
                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 3, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 3, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 3, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::U8 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::U8 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::U8, Precision::FP32 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::I8 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::I8 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },
                        mvn_test_params{{2, 64, 8, 8, 8}, 0, 1, 0.00001f, 1, false, MKLDNNPlugin::impl_desc_type::unknown, Precision::I8, Precision::FP32 },

                        mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 3, true, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                        mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 3, true, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 },
                        mvn_test_params{{2, 64, 24, 32, 40}, 0, 1, 0.00001f, 3, true, MKLDNNPlugin::impl_desc_type::unknown, Precision::FP32, Precision::FP32 }
        ));
