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

struct argmax_test_params {
    std::vector<size_t> src_dims;
    std::vector<size_t> dst_dims;
    int has_axis;
    int axis;
    int out_max_val;
    int top_k;
};

static inline int count(std::vector<size_t> dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

static inline int count(std::vector<size_t> dims, size_t start_ind = 0) {
    return count(dims, start_ind, dims.size());
}

static void ref_argmax(InferenceEngine::TBlob<float> &src, InferenceEngine::TBlob<float> &dst, argmax_test_params p) {
    float *src_data = src.data();
    float* dst_data = dst.data();

    int dim, axis_dist;
    if (p.has_axis) {
        int axis_ = (p.axis < 0) ? p.axis + static_cast<int>(p.src_dims.size()) : p.axis;
        dim = static_cast<int>(p.src_dims[axis_]);
        axis_dist = count(p.src_dims, axis_) / dim;
    } else {
        dim = count(p.src_dims, 1);
        axis_dist = 1;
    }

    int num = count(p.src_dims) / dim;
    std::vector<std::pair<float, int> > src_vector(dim);

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
            src_vector[j] = std::make_pair(
                    src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
        }

        std::partial_sort(src_vector.begin(), src_vector.begin() + p.top_k,
                          src_vector.end(), std::greater<std::pair<float, int> >());

        for (int j = 0; j < p.top_k; ++j) {
            if (p.out_max_val) {
                if (p.has_axis) {
                    // Produces max_val per axis
                    dst_data[(i / axis_dist * p.top_k + j) * axis_dist + i % axis_dist] = src_vector[j].first;
                } else {
                    // Produces max_ind and max_val
                    dst_data[2 * i * p.top_k + j] = src_vector[j].second;
                    dst_data[2 * i * p.top_k + p.top_k + j] = src_vector[j].first;
                }
            } else {
                // Produces max_ind per axis
                dst_data[(i / axis_dist * p.top_k + j) * axis_dist + i % axis_dist] = src_vector[j].second;
            }
        }
    }
}

class smoke_CPU_ArgmaxOnlyTest: public TestsCommon, public WithParamInterface<argmax_test_params> {
    std::string model_t = R"V0G0N(
<net name="ArgmaxOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="input" type="Input" precision="FP32" >
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer id="1" name="argmax" type="ArgMax" precision="FP32">
            <data _AXIS_ out_max_val="__OUT_MAX_VAL__" top_k="__TOP_K__"/>
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
    std::string getModel(argmax_test_params p) {
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

        std::string axis;
        if (p.has_axis) {
            axis += "axis=\"" + std::to_string(p.axis) + "\"";
        }
        REPLACE_WITH_STR(model, "_AXIS_", axis);

        REPLACE_WITH_STR(model, "__OUT_MAX_VAL__", std::to_string(p.out_max_val));
        REPLACE_WITH_STR(model, "__TOP_K__", std::to_string(p.top_k));

        return model;
    }

    virtual void SetUp() {
        try {
            argmax_test_params p = ::testing::WithParamInterface<argmax_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());

            Blob::Ptr src = make_shared_blob<float>({Precision::FP32, p.src_dims, Layout::ANY});
            src->allocate();

            TBlob<float>* srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            CommonTestUtils::fill_data_sine(src->buffer(), src->size(), 0.5, 0.5, 1);

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

            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_argmax(*srcPtr, dst_ref, p);

            ExecutableNetwork exeNetwork = ie.LoadNetwork(net, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(srcs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            compare(*outputBlobs.begin()->second, dst_ref);

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPU_ArgmaxOnlyTest, TestsArgmax) {}

INSTANTIATE_TEST_CASE_P(
        TestsArgmax, smoke_CPU_ArgmaxOnlyTest,
        ::testing::Values(
                argmax_test_params{{1, 3, 1024, 2048}, {1, 1, 1024, 2048}, 1, 1, 0, 1},
                argmax_test_params{{1, 5, 1024, 2048}, {1, 1, 1024, 2048}, 1, 1, 1, 1},
                argmax_test_params{{3, 1, 10, 512}, {3}, 0, 1, 0, 1},
                argmax_test_params{{3, 1, 10, 512}, {3, 2}, 0, 1, 1, 1},
                argmax_test_params{{1, 20, 128, 128}, {1, 3, 128, 128}, 1, 1, 0, 3},
                argmax_test_params{{1, 20, 128, 128}, {1, 3, 128, 128}, 1, 1, 1, 3},
                argmax_test_params{{3, 1, 10, 512}, {3, 5}, 0, 1, 0, 5},
                argmax_test_params{{3, 1, 10, 512}, {3, 5, 2}, 0, 1, 1, 5},
                argmax_test_params{{1, 20, 128, 128}, {1, 18, 128, 128}, 1, 1, 0, 18},
                argmax_test_params{{1, 20, 128, 128}, {1, 18, 128, 128}, 1, 1, 1, 18}
        ));

INSTANTIATE_TEST_CASE_P(
        TestsArgmaxOddDims, smoke_CPU_ArgmaxOnlyTest,
        ::testing::Values(
                argmax_test_params{{1, 3, 1025, 2049}, {1, 1, 1025, 2049}, 1, 1, 0, 1},
                argmax_test_params{{1, 5, 1025, 2049}, {1, 1, 1025, 2049}, 1, 1, 1, 1},
                argmax_test_params{{1, 20, 129, 129}, {1, 3, 129, 129}, 1, 1, 0, 3},
                argmax_test_params{{1, 20, 129, 129}, {1, 3, 129, 129}, 1, 1, 1, 3}
        ));