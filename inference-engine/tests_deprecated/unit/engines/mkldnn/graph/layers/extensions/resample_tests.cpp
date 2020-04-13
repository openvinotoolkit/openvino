// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include "ir_gen_helper.hpp"
#include <cpp/ie_cnn_net_reader.h>

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

struct resample_test_params {
    std::vector<size_t> in_dims;

    float factor;
    int antialias;
    std::string type;

    size_t num_prim_desc;
    bool isBlockedFormat;
    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};


static inline float triangleCoeff(float x) {
    return max(0.0f, 1 - std::abs(x));
}

extern InferenceEngine::IExtensionPtr make_FakeExtensions();

template <typename data_t>
void ref_resample(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, resample_test_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    size_t ndims = prm.in_dims.size();

    size_t N = prm.in_dims[0];
    size_t C = prm.in_dims[1];
    size_t ID = ndims == 5 ? prm.in_dims[ndims - 3] : 1;
    size_t IH = prm.in_dims[ndims - 2];
    size_t IW = prm.in_dims[ndims - 1];
    size_t OD = ndims == 5 ? ID / prm.factor : 1;
    size_t OH = IH / prm.factor;
    size_t OW = IW / prm.factor;

    float fx = static_cast<float>(IW) / static_cast<float>(OW);
    float fy = static_cast<float>(IH) / static_cast<float>(OH);
    float fz = static_cast<float>(ID) / static_cast<float>(OD);

    if (prm.type == "caffe.ResampleParameter.NEAREST") {
        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float *in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float *out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;

                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fx / 2.0f - 0.5f;
                            float iy = oy * fy + fy / 2.0f - 0.5f;
                            float iz = oz * fz + fz / 2.0f - 0.5f;

                            size_t ix_r = static_cast<size_t>(round(ix));
                            size_t iy_r = static_cast<size_t>(round(iy));
                            size_t iz_r = static_cast<size_t>(round(iz));

                            out_ptr[oz * OH * OW + oy * OW + ox] = in_ptr[iz_r * IH * IW + iy_r * IW + ix_r];
                        }
                    }
                }
            }
        }
    } else if (prm.type == "caffe.ResampleParameter.LINEAR") {
        size_t kernel_width = 2;
        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
        bool antialias = isDownsample && prm.antialias;

        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float *in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float *out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;

                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fx / 2.0f - 0.5f;
                            float iy = oy * fy + fy / 2.0f - 0.5f;
                            float iz = oz * fz + fz / 2.0f - 0.5f;

                            int ix_r = static_cast<int>(round(ix));
                            int iy_r = static_cast<int>(round(iy));
                            int iz_r = static_cast<int>(round(iz));

                            float sum = 0;
                            float wsum = 0;

                            float ax = 1.0f / (antialias ? fx : 1.0f);
                            float ay = 1.0f / (antialias ? fy : 1.0f);
                            float az = 1.0f / (antialias ? fz : 1.0f);

                            int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
                            int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
                            int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

                            for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                                for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                                    for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                        if (z < 0 || y < 0 || x < 0 || z >= static_cast<int>(ID) ||y >= static_cast<int>(IH) || x >= static_cast<int>(IW))
                                            continue;

                                        float dx = ix - x;
                                        float dy = iy - y;
                                        float dz = iz - z;

                                        float w = ax * triangleCoeff(ax * dx) * ay * triangleCoeff(ay * dy) * az * triangleCoeff(az * dz);

                                        sum += w * in_ptr[z * IH * IW + y * IW + x];
                                        wsum += w;
                                    }
                                }
                            }
                            out_ptr[oz * OH * OW + oy * OW + ox] = (!wsum) ? 0 : (sum / wsum);
                        }
                    }
                }
            }
        }
    } else {
        assert(!"Unsupported resample operation type");
    }
}

class MKLDNNCPUExtResampleTests: public TestsCommon, public WithParamInterface<resample_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Resample_net" version="2" precision="FP32" batch="1">
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
        <layer name="fakeLayer" id="1" type="_FL_" precision="FP32">
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
        <layer name="resample" id="2" type="Resample" precision="FP32">
            <data antialias="_AN_" factor="_F_" type="_T_"/>
            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_OD_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
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

    std::string getModel(resample_test_params p) {
        std::string model = model_t;

        auto dims_size = p.in_dims.size();
        if (dims_size == 4) {
            REMOVE_LINE(model, "<dim>_ID_</dim>");
            REMOVE_LINE(model, "<dim>_OD_</dim>");
        }

        if (p.isBlockedFormat)
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerBLK");
        else
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerPLN");

        REPLACE_WITH_NUM(model, "_IN_", p.in_dims[0]);
        REPLACE_WITH_NUM(model, "_IC_", p.in_dims[1]);
        if (dims_size == 5)
            REPLACE_WITH_NUM(model, "_ID_", p.in_dims[dims_size - 3]);
        REPLACE_WITH_NUM(model, "_IH_", p.in_dims[dims_size - 2]);
        REPLACE_WITH_NUM(model, "_IW_", p.in_dims[dims_size - 1]);

        if (dims_size == 5)
            REPLACE_WITH_NUM(model, "_OD_", (int)(p.in_dims[dims_size - 3] / p.factor));
        REPLACE_WITH_NUM(model, "_OH_", (int)(p.in_dims[dims_size - 2] / p.factor));
        REPLACE_WITH_NUM(model, "_OW_", (int)(p.in_dims[dims_size - 1] / p.factor));

        REPLACE_WITH_NUM(model, "_AN_", p.antialias);
        REPLACE_WITH_NUM(model, "_F_", p.factor);
        REPLACE_WITH_STR(model, "_T_", p.type);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            resample_test_params p = ::testing::WithParamInterface<resample_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            auto defaultExtensions = std::make_shared<InferenceEngine::Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::isa_any>>();
            extMgr->AddExtension(defaultExtensions);
            extMgr->AddExtension(make_FakeExtensions());

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "resample") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }

            InferenceEngine::SizeVector dims_src = p.in_dims;

            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in_dims.size()) {
                case 4: layout = InferenceEngine::NCHW; break;
                case 5: layout = InferenceEngine::NCDHW; break;
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, layout});
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
            ref_resample(*srcPtr, dst_ref, p);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtResampleTests, TestsResample) {}

INSTANTIATE_TEST_CASE_P(
        TestsResample, MKLDNNCPUExtResampleTests,
        ::testing::Values(
                resample_test_params{{2, 64, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 25}, 1.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 0.25f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 10, 20}, 4.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 25}, 1.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 0.25f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 10, 20}, 4.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                // 5D nearest
                resample_test_params{{2, 64, 20, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 20, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 64, 15, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 20, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 20, 15, 25}, 1.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 3, 15, 10, 20}, 4.f, 0, "caffe.ResampleParameter.NEAREST", 3, true, MKLDNNPlugin::impl_desc_type::unknown },
                // 5D linear
                resample_test_params{{2, 15, 15, 10, 20}, 9.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 15, 15, 10, 20}, 1.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 15, 15, 10, 20}, 4.f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 2, 15, 10, 20}, 0.25f, 1, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 15, 15, 10, 20}, 9.f, 0, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 15, 15, 10, 20}, 1.f, 0, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 8, 15, 10, 20}, 4.f, 0, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown },
                resample_test_params{{2, 2, 15, 10, 20}, 0.25f, 0, "caffe.ResampleParameter.LINEAR", 1, false, MKLDNNPlugin::impl_desc_type::unknown }));