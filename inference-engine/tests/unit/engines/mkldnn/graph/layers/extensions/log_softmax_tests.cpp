// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <extension/ext_list.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct log_softmax_test_params {
    InferenceEngine::SizeVector in_out;
    std::vector<float>          src;
    int                         axis;
    std::vector<float>          reference;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_log_softmax(InferenceEngine::TBlob<float> &src, int axis, InferenceEngine::TBlob<float> &dst) {
    float *src_data = src.data();
    float *dst_data = dst.data();
    InferenceEngine::SizeVector dims = src.getTensorDesc().getDims();

    if (axis < 0) axis += dims.size();

    size_t W = dims[3];
    size_t H = dims[2];
    size_t C = dims[1];
    size_t MB = dims[0];

    auto off = [=](int n, int c, int h, int w)
    {
        return (n * W * H * C + c * W * H + h * W + w);
    };

    if(axis == 0) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float result = 0.0f;
                    for (int n = 0; n < MB; ++n) {
                        result += expf(src_data[off(n, c, h, w)]);
                    }
                    result = logf(result);
                    for (int n = 0; n < MB; ++n) {
                        dst_data[off(n, c, h, w)] = src_data[off(n, c, h, w)] - result;
                    }
                }
            }
        }
    } else if(axis == 1) {
        for (int n = 0; n < MB; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float result = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        result += expf(src_data[off(n, c, h, w)]);
                    }
                    result = logf(result);
                    for (int c = 0; c < C; ++c) {
                        dst_data[off(n, c, h, w)] = src_data[off(n, c, h, w)] - result;
                    }
                }
            }
        }
    } else if(axis == 2) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int w = 0; w < W; ++w) {
                    float result = 0.0f;
                    for (int h = 0; h < H; ++h) {
                        result += expf(src_data[off(n, c, h, w)]);
                    }
                    result = logf(result);
                    for (int h = 0; h < H; ++h) {
                        dst_data[off(n, c, h, w)] = src_data[off(n, c, h, w)] - result;
                    }
                }
            }
        }
    } else if(axis == 3) {
        for (int n = 0; n < MB; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    float result = 0.0f;
                    for (int w = 0; w < W; ++w) {
                        result += expf(src_data[off(n, c, h, w)]);
                    }
                    result = logf(result);
                    for (int w = 0; w < W; ++w) {
                        dst_data[off(n, c, h, w)] = src_data[off(n, c, h, w)] - result;
                    }
                }
            }
        }
    }
}

void ref_log_softmax_any_dims(InferenceEngine::TBlob<float> &src, int axis, InferenceEngine::TBlob<float> &dst) {
    size_t i, j, k, axis_step = 1, reduced_axis_size, reduced_axis_stride = 1;
    InferenceEngine::SizeVector dims = src.getTensorDesc().getDims();
    float *src_data = src.data();
    float *dst_data = dst.data();

    if (axis < 0) axis += dims.size();
    for (i = 0; i < axis; i++) axis_step *= dims[i];
    reduced_axis_size = dims[axis];
    for (i = (axis + 1); i < dims.size(); i++) reduced_axis_stride *= dims[i];

    for (k = 0; k < axis_step; k++) {
        for (i = 0; i < reduced_axis_stride; i++) {
            float reduce_prod = 0.0f;
            const float *src_dataPtr = &src_data[k * reduced_axis_stride * reduced_axis_size + i];
            for (j = 0; j < reduced_axis_size; ++j) {
                reduce_prod += expf((*src_dataPtr));
                src_dataPtr += reduced_axis_stride;
            }

            reduce_prod = logf(reduce_prod);
            src_dataPtr = &src_data[k * reduced_axis_stride * reduced_axis_size + i];
            float *dst_dataPtr = (float*)&dst_data[k * reduced_axis_stride * reduced_axis_size + i];
            for (j = 0; j < reduced_axis_size; ++j) {
                (*dst_dataPtr) = (*src_dataPtr) - reduce_prod;
                src_dataPtr += reduced_axis_stride;
                dst_dataPtr += reduced_axis_stride;
            }
        }
    }
}

class MKLDNNCPUExtLogSoftmaxTests : public TestsCommon, public WithParamInterface<log_softmax_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Math_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="Input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="math" id="2" type="LogSoftmax" precision="FP32">
            <data axis="_AXIS_"/>
            <input>
                <port id="1">
                    _IN_OUT_
                </port>
            </input>
            <output>
                <port id="3">
                    _IN_OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(log_softmax_test_params p) {
        std::string model = model_t;
        std::string in_out;

        for (auto& dst : p.in_out) {
            in_out += "<dim>";
            in_out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IN_OUT_", in_out);
        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            log_softmax_test_params p = ::testing::WithParamInterface<log_softmax_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            // Input Data
            InferenceEngine::Blob::Ptr srcData = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_out, InferenceEngine::TensorDesc::getLayoutByDims(p.in_out) });
            srcData->allocate();
            if (p.src.size())
                memcpy(srcData->buffer(), &p.src[0], sizeof(float)*p.src.size());
            else
                fill_data(srcData->buffer(), srcData->size());
            auto * srcDataPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcData.get());
            if (srcDataPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Check results
            if (p.in_out.size() == 4) {
                ref_log_softmax(*srcDataPtr, p.axis, dst_ref);
                if (p.reference.size()) {
                    for (size_t i = 0; i < p.reference.size(); i++) {
                        ASSERT_NEAR(dst_ref.data()[i], p.reference[i], 0.00001f);
                    }
                }
            }
            ref_log_softmax_any_dims(*srcDataPtr, p.axis, dst_ref);
            if (p.reference.size()) {
                for (size_t i = 0; i < p.reference.size(); i++) {
                    ASSERT_NEAR(dst_ref.data()[i], p.reference[i], 0.00001f);
                }
            }

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("Input", srcData));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref, 0.00001f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtLogSoftmaxTests, TestsLogSoftmax) {}

INSTANTIATE_TEST_CASE_P(
    TestsLogSoftmax, MKLDNNCPUExtLogSoftmaxTests,
        ::testing::Values(
            // Params: in_out, src, axis, reference
            log_softmax_test_params{ { 1, 1, 1, 3 },{ -0.5f, 0.f, 0.5f },3,{ -1.68026966f, -1.1802697f, -0.68026966 } },
            log_softmax_test_params{ { 1, 1, 1, 3 },{ -0.5f, 0.f, 0.5f },-1,{ -1.68026966f, -1.1802697f, -0.68026966 } },
            log_softmax_test_params{ { 3, 1, 1, 1 },{ -0.5f, 0.f, 0.5f },0,{ -1.68026966f, -1.1802697f, -0.68026966 } },
            log_softmax_test_params{ { 1, 1, 2, 2 },{ 1.0f, 0.5f, 0.f, -0.5f },3,{ -0.474077f, -0.974077f, -0.474077f, -0.974077f } },
            log_softmax_test_params{ { 2, 2, 1, 1 },{ 1.0f, 0.5f, 0.f, -0.5f },1,{ -0.474077f, -0.974077f, -0.474077f, -0.974077f } },
            log_softmax_test_params{ { 2, 2, 1, 1 },{ 1.0f, 0.5f, 0.f, -0.5f },-3,{ -0.474077f, -0.974077f, -0.474077f, -0.974077f } },
            log_softmax_test_params{ { 2, 3, 3, 2 },{ },3,{ } },
            log_softmax_test_params{ { 1, 1, 2, 2 },{ 1.0f, 0.5f, 0.f, -0.5f },2,{ -0.31326166f, -0.31326166f, -1.3132616f, -1.3132616f } },
            log_softmax_test_params{ { 2, 3, 3, 2 },{},0,{} },
            log_softmax_test_params{ { 2, 3, 3, 2 },{},1,{} },
            log_softmax_test_params{ { 2, 3, 3, 2 },{},2,{} },
            log_softmax_test_params{ { 2, 3, 3, 2, 4, 5, 1, 2 },{},4,{} }
        ));
