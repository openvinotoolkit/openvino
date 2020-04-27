// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cmath>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;

struct resample_test_params {
    std::string device_name;
    InferenceEngine::SizeVector in_dims;
    float factor;
    std::string type;
};

static inline float triangleCoeff(float x) {
    return std::max(0.0f, 1 - std::abs(x));
}

template <typename data_t>
static void ref_resample(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst, resample_test_params p) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    size_t ndims = p.in_dims.size();

    size_t N = p.in_dims[0];
    size_t C = p.in_dims[1];
    size_t ID = ndims == 5 ? p.in_dims[ndims - 3] : 1;
    size_t IH = p.in_dims[ndims - 2];
    size_t IW = p.in_dims[ndims - 1];
    size_t OD = ndims == 5 ? static_cast<size_t>(ID / p.factor) : 1;
    size_t OH = static_cast<size_t>(IH / p.factor);
    size_t OW = static_cast<size_t>(IW / p.factor);

    float fx = static_cast<float>(IW) / static_cast<float>(OW);
    float fy = static_cast<float>(IH) / static_cast<float>(OH);
    float fz = static_cast<float>(ID) / static_cast<float>(OD);

    if (p.type == "caffe.ResampleParameter.NEAREST") {
        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float* in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float* out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;
                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx;
                            float iy = oy * fy;
                            float iz = oz * fz;

                            size_t ix_r = static_cast<size_t>(std::floor(ix));
                            size_t iy_r = static_cast<size_t>(std::floor(iy));
                            size_t iz_r = static_cast<size_t>(std::floor(iz));

                            out_ptr[oz * OH * OW + oy * OW + ox] = in_ptr[iz_r * IH * IW + iy_r * IW + ix_r];
                        }
                    }
                }
            }
        }
    } else if (p.type == "caffe.ResampleParameter.LINEAR") {
        size_t kernel_width = 2;
        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
        bool antialias = false;

        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                const float* in_ptr = src_data + IW * IH * ID * C * b + IW * IH * ID * c;
                float* out_ptr = dst_data + OW * OH * OD * C * b + OW * OH * OD * c;

                for (size_t oz = 0; oz < OD; oz++) {
                    for (size_t oy = 0; oy < OH; oy++) {
                        for (size_t ox = 0; ox < OW; ox++) {
                            float ix = ox * fx + fy / 2.0f - 0.5f;
                            float iy = oy * fy + fx / 2.0f - 0.5f;
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
                                        if (z < 0 || y < 0 || x < 0 || z >= static_cast<int>(ID) || y >= static_cast<int>(IH) || x >= static_cast<int>(IW))
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

class ResampleTests : public TestsCommon, public WithParamInterface<resample_test_params> {
    std::string model_t = R"V0G0N(
<net Name="resample_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
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
                    <dim>_OD_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(resample_test_params p) {
        std::string model = model_t;
        std::string inDim;

        auto dims_size = p.in_dims.size();
        if (dims_size == 4) {
            REMOVE_LINE(model, "<dim>_ID_</dim>");
            REMOVE_LINE(model, "<dim>_OD_</dim>");
        }

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

        REPLACE_WITH_NUM(model, "_AN_", 0);
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

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in_dims.size()) {
            case 4: layout = InferenceEngine::NCHW; break;
            case 5: layout = InferenceEngine::NCDHW; break;
            default:
                FAIL() << "Input dims size not supported in this test.";
            }

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.in_dims, layout});
            src->allocate();
            fill_data(src->buffer(), src->size());
            for (size_t i = 0; i < src->size(); i++) {
                src->buffer().as<float*>()[i] = static_cast<float>(i);
            }

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

            InferenceEngine::OutputsDataMap out;
            out = net.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_resample<float>(*srcPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(ResampleTests, TestsResample) {}
