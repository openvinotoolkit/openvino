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


struct interp_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    struct {
        size_t h;
        size_t w;
    } out;

    int pad_beg;
    int pad_end;

    size_t num_prim_desc;

    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void interpolate(const int N, const int C, const float *src, const int x1, const int y1, const int IH_pad, const int IW_pad,
                      const int IH, const int IW, float *dst, const int x2, const int y2, const int OH_pad, const int OW_pad, const int OH, const int OW) {
    if (IH_pad == OH_pad && IW_pad == OW_pad) {
        for (int i = 0; i < N * C * OH * OW; i++) {
            dst[i] = src[i];
        }
        return;
    }

    const float rh = (OH_pad > 1) ? static_cast<float>(IH_pad - 1) / (OH_pad - 1) : 0.0f;
    const float rw = (OW_pad > 1) ? static_cast<float>(IW_pad - 1) / (OW_pad - 1) : 0.0f;

    const int block_size = 1;

    // Align channel number to block size to deal with channels padding in IE with multiple blobs
    int CB = (C + block_size - 1) & (-block_size);

    int CH = (C + block_size - 1) / block_size;

    for (int n = 0; n < N; n++) {
        for (int cb = 0; cb < CH; ++cb) {
            for (int h = 0; h < OH_pad; ++h) {
                const float *psrc = src + n * CB * IH * IW;

                float fh = rh * h;
                int ih0 = static_cast<int>(fh);
                int ih1 = (ih0 < IH_pad - 1) ? ih0 + 1 : ih0;

                float h_lambda0 = fh - ih0;
                float h_lambda1 = 1.0f - h_lambda0;

                for (int w = 0; w < OW_pad; ++w) {
                    float fw = rw * w;
                    int iw0 = static_cast<int>(fw);
                    int iw1 = (iw0 < IW_pad - 1) ? iw0 + 1 : iw0;

                    float w_lambda0 = fw - iw0;
                    float w_lambda1 = 1.0f - w_lambda0;

                    const float *psrc00 =
                            psrc + cb * block_size * IW * IH + (y1 + ih0) * IW * block_size + (x1 + iw0) * block_size;
                    const float *psrc01 =
                            psrc + cb * block_size * IW * IH + (y1 + ih0) * IW * block_size + (x1 + iw1) * block_size;
                    const float *psrc10 =
                            psrc + cb * block_size * IW * IH + (y1 + ih1) * IW * block_size + (x1 + iw0) * block_size;
                    const float *psrc11 =
                            psrc + cb * block_size * IW * IH + (y1 + ih1) * IW * block_size + (x1 + iw1) * block_size;

                    float *pdst = dst + n * CB * OH * OW + cb * block_size * OW * OH + (y2 + h) * OW * block_size +
                                  (x2 + w) * block_size;

                    for (int c = 0; c < block_size; ++c) {
                        pdst[c] = h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                                  h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
                    }
                }
            }
        }
    }
}

template <typename data_t>
void ref_interp(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, interp_test_params prm) {
    int IB = static_cast<int>(src.getTensorDesc().getDims()[0]);
    int IC = static_cast<int>(src.getTensorDesc().getDims()[1]);
    int IH = static_cast<int>(src.getTensorDesc().getDims()[2]);
    int IW = static_cast<int>(src.getTensorDesc().getDims()[3]);

    int OH = static_cast<int>(dst.getTensorDesc().getDims()[2]);
    int OW = static_cast<int>(dst.getTensorDesc().getDims()[3]);

    int IH_pad = IH + prm.pad_beg + prm.pad_end;
    int IW_pad = IW + prm.pad_beg + prm.pad_end;

    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    interpolate(IB, IC, src_data, -prm.pad_beg, -prm.pad_beg, IH_pad, IW_pad, IH, IW, dst_data, 0, 0, OH, OW, OH, OW);
}

class MKLDNNCPUExtInterpTests: public TestsCommon, public WithParamInterface<interp_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
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
        <layer name="interp1" id="1" type="Interp" precision="FP32">
            <data pad_beg="_PB_" pad_end="_PE_"/>

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

    std::string getModel(interp_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        REPLACE_WITH_NUM(model, "_PB_", p.pad_beg);
        REPLACE_WITH_NUM(model, "_PE_", p.pad_end);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            interp_test_params p = ::testing::WithParamInterface<interp_test_params>::GetParam();
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
                if (node->getName() == "interp1") {
                    ASSERT_LE(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            ASSERT_LE(4, nodes.size());

            InferenceEngine::SizeVector dims_src = {p.in.w, p.in.h, p.in.c, p.in.n};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
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
            ref_interp(*srcPtr, dst_ref, p);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtInterpTests, TestsInterp) {}

INSTANTIATE_TEST_CASE_P(
        TestsInterp, MKLDNNCPUExtInterpTests,
        ::testing::Values(
                interp_test_params{{1, 256, 1, 1}, {33, 65}, 0, 0, 1, MKLDNNPlugin::impl_desc_type::unknown },
                interp_test_params{{1, 2, 33, 65}, {33, 65}, 0, 0, 1, MKLDNNPlugin::impl_desc_type::unknown }));
