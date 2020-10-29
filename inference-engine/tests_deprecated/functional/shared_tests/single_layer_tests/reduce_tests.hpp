// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ie_memcpy.h"

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;

struct reduce_test_params {
    std::string                 device_name;
    std::string                 inIdxPrecision;;
    std::string                 reduce_type;
    bool                        keep_dims;
    SizeVector in_shape;
    std::vector<float>          input_tensor;
    std::vector<int32_t>        axes_for_reduction;
    SizeVector out_shape;
    std::vector<float>          reference;
};

template <typename F>
void reduce(
        const float* src_data,
        SizeVector src_dims,
        SizeVector srcStrides,
        float* dst_data,
        SizeVector dst_dims,
        SizeVector dstStrides,
        float init_value,
        bool keep_dims,
        SizeVector skip_dims,
        F func
) {
    size_t i, src_idx, dst_idx;
    for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
        dst_data[i] = init_value;

    SizeVector counters(src_dims.size(), 0);
    for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx) {
        if (keep_dims)
            for (i = 0, dst_idx = 0; i < dst_dims.size(); ++i)
                dst_idx += (counters[i] % dst_dims[i]) * dstStrides[i];
        else
            for (i = 0, dst_idx = 0; i < dst_dims.size(); ++i)
                dst_idx += counters[skip_dims[i]] * dstStrides[i];

        dst_data[dst_idx] = func(dst_data[dst_idx], src_data[src_idx]);
        for (int j = src_dims.size() - 1; j >= 0; j--) {
            counters[j] = (counters[j] + 1) % src_dims[j];
            if (counters[j] != 0) break;
        }
    }
}

void ref_reduce(
    std::string reduce_type,
    TBlob<float> &src,
    bool keep_dims,
    std::vector<int32_t> axes_for_reduction,
    TBlob<float> &dst,
    SizeVector &out_dims
) {
    size_t i, src_idx, dst_idx;
    const float* src_data = src.data();
    SizeVector src_dims = src.getTensorDesc().getDims();
    SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    float* dst_data = dst.data();
    SizeVector dst_dims = dst.getTensorDesc().getDims();
    SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();
    SizeVector skip_dims;

    if (!dst_dims.size())
        dst_dims = InferenceEngine::SizeVector(1, 1);

    if (!dstStrides.size())
        dstStrides = InferenceEngine::SizeVector(1, 1);

    if (axes_for_reduction.size() == 0)
        FAIL() << " Index vector should be 1 dimension";

    for (i = 0; i < axes_for_reduction.size(); i++) {
        int32_t axis = axes_for_reduction[i];
        if (axis < 0)
            axis += src_dims.size();

        if (axis > src_dims.size())
            FAIL() << " Index to squeeze exceeds data tensor dimension";
        axes_for_reduction[i] = axis;
    }

    for (size_t j = 0; j < src_dims.size(); j++) {
        bool found = false;
        for (size_t axis : axes_for_reduction)
            if (j == axis) found = true;

        if (!found) {
            out_dims.push_back(src_dims[j]);
            if (!keep_dims) skip_dims.push_back(j);
        }
        else {
            if (keep_dims) out_dims.push_back(1);
        }
    }

    if (reduce_type == "ReduceAnd") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 1.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x && y; } );
        } else {
            dst_data[0] = 1.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] && src_data[src_idx];
        }
    } else if (reduce_type == "ReduceL1") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + (std::abs)(y); } );
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += (std::abs)(src_data[src_idx]);
        }
    } else if (reduce_type == "ReduceL2") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + y * y; } );

            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] = (std::sqrt)(dst_data[i]);
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx] * src_data[src_idx];
            dst_data[0] = sqrt(dst_data[0]);
        }
    } else if (reduce_type == "ReduceLogSum") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + y; });

            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] = logf(dst_data[i]);
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
            dst_data[0] = logf(dst_data[0]);
        }
    } else if (reduce_type == "ReduceLogSumExp") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + expf(y); });

            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] = logf(dst_data[i]);
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += expf(src_data[src_idx]);
            dst_data[0] = logf(dst_data[0]);
        }
    } else if (reduce_type == "ReduceMax") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, FLT_MIN, keep_dims, skip_dims,
                   [](float x, float y)->float { return x > y ? x : y; });
        } else {
            dst_data[0] = FLT_MIN;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] > src_data[src_idx] ? dst_data[0] : src_data[src_idx];
        }
    } else if (reduce_type == "ReduceMean") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + y; });
            float reduced_dims_work_amount = 1.f;
            for (size_t axis : axes_for_reduction) {
                reduced_dims_work_amount *= static_cast<float>(src_dims[axis]);
            }
            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] /= reduced_dims_work_amount;
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
            dst_data[0] /= static_cast<float>(srcStrides[0] * src_dims[0]);
        }
    } else if (reduce_type == "ReduceMin") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, FLT_MAX, keep_dims, skip_dims,
                   [](float x, float y)->float { return x < y ? x : y; });
        } else {
            dst_data[0] = FLT_MAX;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] < src_data[src_idx] ? dst_data[0] : src_data[src_idx];
        }
    } else if (reduce_type == "ReduceOr") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x || y; });
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] || src_data[src_idx];
        }
    } else if (reduce_type == "ReduceProd") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 1.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x * y; });
        } else {
            dst_data[0] = 1.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] *= src_data[src_idx];
        }
    } else if (reduce_type == "ReduceSum") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + y; });
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
        }
    } else if (reduce_type == "ReduceSumSquare") {
        if (out_dims.size()) {
            reduce(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0.0f, keep_dims, skip_dims,
                   [](float x, float y)->float { return x + y * y; });
        } else {
            dst_data[0] = 0.0f;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx] * src_data[src_idx];
        }
    }
}

class ReduceTestsShared : public TestsCommon, public WithParamInterface<reduce_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Reduce_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer id="2" name="input2" precision="_IIDXP_" type="Const">
            <output>
                <port id="1">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="_DIM_SIZE_"/>
            </blobs>
        </layer>
        <layer name="reduce_REDUCE_TYPE_" id="3" type="_REDUCE_TYPE_" precision="FP32">
            <data keep_dims="_KEEP_DIMS_" />
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(reduce_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape = "";

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.axes_for_reduction.size());
        REPLACE_WITH_STR(model, "_REDUCE_TYPE_", p.reduce_type);
        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_NUM(model, "_KEEP_DIMS_", p.keep_dims);

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    static void fill_data_dbgval(float* data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i + 1;
        }
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            reduce_test_params p = ::testing::WithParamInterface<reduce_test_params>::GetParam();
            std::string model = getModel(p);

            // std::cout << model << std::endl;

            TBlob<uint8_t> * axes = nullptr;
            if (p.inIdxPrecision == "I32") {
                axes = new TBlob<uint8_t>({Precision::U8,
                    {p.axes_for_reduction.size() * sizeof(int32_t)},
                    Layout::C});
                axes->allocate();
                for (size_t i = 0; i < p.axes_for_reduction.size(); i++) {
                    ((int32_t *) axes->buffer())[i] = p.axes_for_reduction[i];
                }
            } else {
                axes = new TBlob<uint8_t>({Precision::U8,
                    { p.axes_for_reduction.size() * sizeof(float) },
                    Layout::C});
                axes->allocate();
                for (size_t i = 0; i < p.axes_for_reduction.size(); i++) {
                    ((float *) axes->buffer())[i] = p.axes_for_reduction[i];
                }
            }
            
            Core ie;
            auto net = ie.ReadNetwork(model, TBlob<uint8_t>::Ptr(axes));
            OutputsDataMap out = net.getOutputsInfo();
            std::pair<std::string, DataPtr> item = *out.begin();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Input Data
            Blob::Ptr src;
            src = make_shared_blob<float>({ Precision::FP32, p.in_shape, TensorDesc::getLayoutByDims(p.in_shape) });
            src->allocate();
            if(p.input_tensor.size())
                ie_memcpy(src->buffer(), src->byteSize(), &p.input_tensor[0], sizeof(float)*p.input_tensor.size());
            else
                fill_data_dbgval(src->buffer(), src->size());

            auto* srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Output Reference
            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            SizeVector out_dims;
            ref_reduce(p.reduce_type, *srcPtr, p.keep_dims, p.axes_for_reduction, dst_ref, out_dims);
            if (p.out_shape.size()>0 && out_dims.size() != p.out_shape.size())
                FAIL() << "Wrong out_shape size!";
            for (size_t i = 0; i < p.out_shape.size(); i++) {
                if (out_dims[i] != p.out_shape[i])
                    FAIL() << "Wrong out_shape dimensions!";
            }
            if (p.reference.size())
                if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                    FAIL() << "Wrong result with compare reference vector!";

            // Output Data
            auto output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);

            // Input
            inferRequest.SetBlob("input", src);
            inferRequest.Infer();

            compare(*output, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(ReduceTestsShared, SharedReduceTests) {}
