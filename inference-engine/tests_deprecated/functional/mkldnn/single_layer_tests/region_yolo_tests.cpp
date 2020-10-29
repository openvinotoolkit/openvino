// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct region_yolo_test_params {
    std::vector<size_t> src_dims;
    std::vector<size_t> dst_dims;
    int classes;
    int coords;
    int num;
    float do_softmax;
    std::vector<int> mask;
};

static inline int entry_index(int width, int height, int coords, int classes, int outputs, int batch, int location,
                       int entry) {
    int n = location / (width * height);
    int loc = location % (width * height);
    return batch * outputs + n * width * height * (coords + classes + 1) +
           entry * width * height + loc;
}

static inline float logistic_activate(float x) {
    return 1.f / (1.f + exp(-x));
}

static inline
void softmax_generic(const float *src_data, float *dst_data, int B, int C, int H, int W) {
    int start = 0;
    for (int b = 0; b < B; b++) {
        for (int i = start; i < H * W; i++) {
            float max = src_data[b * C * H * W + i];
            for (int c = 0; c < C; c++) {
                float val = src_data[b * C * H * W + c * H * W + i];
                if (val > max) max = val;
            }

            float expSum = 0;
            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + i] = exp(src_data[b * C * H * W + c * H * W + i] - max);
                expSum += dst_data[b * C * H * W + c * H * W + i];
            }

            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + i] = dst_data[b * C * H * W + c * H * W + i] / expSum;
            }
        }
    }
}

static void ref_region_yolo(InferenceEngine::TBlob<float> &src, InferenceEngine::TBlob<float> &dst, region_yolo_test_params p) {
    float* src_data = src.data();
    float* dst_data = dst.data();

    int mask_size = p.mask.size();;

    int IW = (src.getTensorDesc().getDims().size() > 3) ? src.getTensorDesc().getDims()[3] : 1;
    int IH = (src.getTensorDesc().getDims().size() > 2) ? src.getTensorDesc().getDims()[2] : 1;
    int IC = (src.getTensorDesc().getDims().size() > 1) ? src.getTensorDesc().getDims()[1] : 1;
    int B = (src.getTensorDesc().getDims().size() > 0) ? src.getTensorDesc().getDims()[0] : 1;

    for (int i = 0; i < src.size(); i++) {
        dst_data[i] = src_data[i];
    }

    int end_index = 0;
    int num_ = 0;
    if (p.do_softmax) {
        // Region layer (Yolo v2)
        end_index = IW * IH;
        num_ = p.num;
    } else {
        // Yolo layer (Yolo v3)
        end_index = IW * IH * (p.classes + 1);
        num_ = mask_size;
    }
    int inputs_size = IH * IW * num_ * (p.classes + p.coords + 1);

    for (int b = 0; b < B; b++) {
        for (int n = 0; n < num_; n++) {
            int index = entry_index(IW, IH, p.coords, p.classes, inputs_size, b, n * IW * IH, 0);
            for (int i = index; i < index + 2 * IW * IH; i++) {
                dst_data[i] = logistic_activate(dst_data[i]);
            }

            index = entry_index(IW, IH, p.coords, p.classes, inputs_size, b, n * IW * IH, p.coords);
            for (int i = index; i < index + end_index; i++) {
                dst_data[i] = logistic_activate(dst_data[i]);
            }
        }
    }

    if (p.do_softmax) {
        int index = entry_index(IW, IH, p.coords, p.classes, inputs_size, 0, 0, p.coords + 1);
        int batch_offset = inputs_size / p.num;
        for (int b = 0; b < B * p.num; b++)
            softmax_generic(src_data + index + b * batch_offset, dst_data + index + b * batch_offset, 1, p.classes,
                            IH, IW);
    }
}

class smoke_CPU_RegionYoloOnlyTest: public TestsCommon, public WithParamInterface<region_yolo_test_params> {
    std::string model_t = R"V0G0N(
<net name="RegionYoloOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="input" type="Input" precision="FP32" >
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer id="1" name="region_yolo" type="RegionYolo" precision="FP32">
            <data classes="_CLASSES_" coords="_COORDS_" do_softmax="_DO_SOFTMAX_" mask="_MASK_" num="_NUM_"/>
            <input>
                <port id="0">__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="1">__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(region_yolo_test_params p) {
        std::string model = model_t;


        std::string src_dims;
        for (auto &dim : p.src_dims) {
            src_dims += "\n                    <dim>";
            src_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);

        std::string dst_dims;
        for (auto &dim : p.dst_dims) {
            dst_dims += "\n                    <dim>";
            dst_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__DST_DIMS__", dst_dims);

        std::string mask;
        for (auto &n : p.mask) {
            mask += std::to_string(n) + ",";
        }
        mask.pop_back();
        REPLACE_WITH_STR(model, "_MASK_", mask);


        REPLACE_WITH_STR(model, "_CLASSES_", std::to_string(p.classes));
        REPLACE_WITH_STR(model, "_COORDS_", std::to_string(p.coords));
        REPLACE_WITH_STR(model, "_DO_SOFTMAX_", std::to_string(p.do_softmax));
        REPLACE_WITH_STR(model, "_NUM_", std::to_string(p.num));


        return model;
    }

    virtual void SetUp() {
        try {
            region_yolo_test_params p = ::testing::WithParamInterface<region_yolo_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());

            Blob::Ptr src = make_shared_blob<float>({Precision::FP32, p.src_dims, Layout::ANY});
            src->allocate();

            TBlob<float>* srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            CommonTestUtils::fill_data_sine(src->buffer(), src->size(), 10, 30, 1);

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("input", src));

            OutputsDataMap out;
            out = net.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_region_yolo(*srcPtr, dst_ref, p);

            ExecutableNetwork exeNetwork = ie.LoadNetwork(net, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            compare(*outputBlobs.begin()->second, dst_ref);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPU_RegionYoloOnlyTest, TestsRegionYolo) {}

INSTANTIATE_TEST_CASE_P(
        TestsRegionYolo, smoke_CPU_RegionYoloOnlyTest,
        ::testing::Values(
                region_yolo_test_params{{1, 255, 52, 52}, {1, 255, 52, 52}, 80, 4, 9, 0, {0, 1, 2}},
                region_yolo_test_params{{1, 255, 26, 26}, {1, 255, 26, 26}, 80, 4, 9, 0, {3, 4, 5}},
                region_yolo_test_params{{1, 255, 13, 13}, {1, 255, 13, 13}, 80, 4, 9, 0, {6, 7, 8}},
                region_yolo_test_params{{1, 125, 13, 13}, {1, 21125}, 20, 4, 5, 1, {0, 1, 2}}
        ));
