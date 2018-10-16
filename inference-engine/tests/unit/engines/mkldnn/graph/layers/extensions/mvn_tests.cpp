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
#include <extension/ext_list.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct mvn_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    int across_channels;
    int normalize_variance;
    float eps;

    size_t num_prim_desc;
    bool isBlockedFormat;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_mvn(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, mvn_test_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    size_t N = prm.in.n;
    size_t C = prm.in.c;
    size_t H = prm.in.h;
    size_t W = prm.in.w;

    float eps = prm.eps;

    for (int b = 0; b < N; b++) {
        // Calculate mean value
        if (prm.across_channels) {
            double mean = 0;
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        mean += src_data[b*C*H*W + c*H*W + h*W + w];
                    }
                }
            }
            mean /= C*H*W;
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] - mean;
                    }
                }
            }
        } else {
            for (int c = 0; c < C; c++) {
                double mean = 0;
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        mean += src_data[b*C*H*W + c*H*W + h*W + w];
                    }
                }
                mean /= H*W;

                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] - mean;
                    }
                }
            }
        }
    }

    if (prm.normalize_variance) {
        for (int b = 0; b < N; b++) {
            // Calculate variances value
            if (prm.across_channels) {
                double variance = 0;
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            variance += std::pow(dst_data[b*C*H*W + c*H*W + h*W + w], 2);
                        }
                    }
                }
                variance /= C*H*W;
                variance = std::pow(variance, 0.5f);
                variance += eps;
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dst_data[b*C*H*W + c*H*W + h*W + w] /= variance;
                        }
                    }
                }
            } else {
                for (int c = 0; c < C; c++) {
                    double variance = 0;
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            variance += std::pow(dst_data[b*C*H*W + c*H*W + h*W + w], 2);
                        }
                    }
                    variance /= H*W;
                    variance = std::pow(variance, 0.5f);
                    variance += eps;
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dst_data[b*C*H*W + c*H*W + h*W + w] /= variance;
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNCPUExtMVNTests: public TestsCommon, public WithParamInterface<mvn_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="MVN_net" version="2" precision="FP32" batch="1">
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
        <layer name="mvn" id="2" type="MVN" precision="FP32">
            <data across_channels="_AC_" normalize_variance="_NV_" eps="_EPS_"/>
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

    std::string getModel(mvn_test_params p) {
        std::string model = model_t;
        if (p.isBlockedFormat)
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerBLK");
        else
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerPLN");

        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_AC_", p.across_channels);
        REPLACE_WITH_NUM(model, "_NV_", p.normalize_variance);
        REPLACE_WITH_NUM(model, "_EPS_", p.eps);

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

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            std::shared_ptr<InferenceEngine::IExtension> cpuExt(new InferenceEngine::Extensions::Cpu::CpuExtensions());
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(cpuExt);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

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
            if (p.isBlockedFormat)
                ASSERT_EQ(6, nodes.size());
            else
                ASSERT_EQ(5, nodes.size()); // TODO: should be 4 (redudant reorder in case of both layers are inplace)

            InferenceEngine::SizeVector dims_src = {p.in.w, p.in.h, p.in.c, p.in.n};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NHWC, dims_src);
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

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_mvn(*srcPtr, dst_ref, p);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtMVNTests, TestsMVN) {}

INSTANTIATE_TEST_CASE_P(
        TestsMVN, MKLDNNCPUExtMVNTests,
        ::testing::Values(
                mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown }));
