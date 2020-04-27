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
#include <ie_core.hpp>


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct reduce_test_params {
    std::string                 reduce_type;
    bool                        keep_dims;
    InferenceEngine::SizeVector in_shape;
    std::string                 inType;
    std::vector<float>          input_tensor;
    std::vector<int32_t>        axes_for_reduction;
    InferenceEngine::SizeVector out_shape;
    std::vector<float>          reference;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename src_t, typename dst_t, typename F>
void reduce(
    const src_t *src_data,
    InferenceEngine::SizeVector src_dims,
    InferenceEngine::SizeVector srcStrides,
    dst_t* dst_data,
    InferenceEngine::SizeVector dst_dims,
    InferenceEngine::SizeVector dstStrides,
    dst_t init_value,
    bool keep_dims,
    InferenceEngine::SizeVector skip_dims,
    F func
) {
    size_t i, src_idx, dst_idx;
    for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
        dst_data[i] = init_value;

    InferenceEngine::SizeVector counters(src_dims.size(), 0);
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

template <typename src_t, typename dst_t>
void ref_reduce(
    std::string reduce_type,
    InferenceEngine::TBlob<src_t> &src,
    bool keep_dims,
    std::vector<int32_t> axes_for_reduction,
    InferenceEngine::TBlob<dst_t> &dst,
    InferenceEngine::SizeVector &out_dims
) {
    size_t i, src_idx, dst_idx;
    const src_t *src_data = src.data();
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    dst_t* dst_data = dst.data();
    InferenceEngine::SizeVector dst_dims = dst.getTensorDesc().getDims();
    InferenceEngine::SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();
    InferenceEngine::SizeVector skip_dims;

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
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 1, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x && y; } );
        } else {
            dst_data[0] = 1;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] && src_data[src_idx];
        }
    } else if (reduce_type == "ReduceL1") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                   [](dst_t x, src_t y)->dst_t { return x + (std::abs)(y); } );
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += (std::abs)(src_data[src_idx]);
        }
    } else if (reduce_type == "ReduceL2") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + y * y; } );

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
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + y; });

            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] = logf(dst_data[i]);
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
            dst_data[0] = logf(dst_data[0]);
        }
    } else if (reduce_type == "ReduceLogSumExp") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + expf(y); });

            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] = logf(dst_data[i]);
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += expf(src_data[src_idx]);
            dst_data[0] = logf(dst_data[0]);
        }
    } else if (reduce_type == "ReduceMax") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, (std::numeric_limits<dst_t>::min)(), keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x > y ? x : y; });
        } else {
            dst_data[0] = (std::numeric_limits<dst_t>::min)();
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] > src_data[src_idx] ? dst_data[0] : src_data[src_idx];
        }
    } else if (reduce_type == "ReduceMean") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + y; });
            float reduced_dims_work_amount = 1.f;
            for (size_t axis : axes_for_reduction) {
                reduced_dims_work_amount *= static_cast<float>(src_dims[axis]);
            }
            for (i = 0; i < dstStrides[0] * dst_dims[0]; ++i)
                dst_data[i] /= reduced_dims_work_amount;
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
            dst_data[0] /= static_cast<float>(srcStrides[0] * src_dims[0]);
        }
    } else if (reduce_type == "ReduceMin") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, (std::numeric_limits<dst_t>::max)(), keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x < y ? x : y; });
        } else {
            dst_data[0] = (std::numeric_limits<dst_t>::max)();
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] < src_data[src_idx] ? dst_data[0] : src_data[src_idx];
        }
    } else if (reduce_type == "ReduceOr") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                   [](dst_t x, src_t y)->dst_t { return x || y; });
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] = dst_data[0] || src_data[src_idx];
        }
    } else if (reduce_type == "ReduceProd") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 1, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x * y; });
        } else {
            dst_data[0] = 1;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] *= src_data[src_idx];
        }
    } else if (reduce_type == "ReduceSum") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + y; });
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx];
        }
    } else if (reduce_type == "ReduceSumSquare") {
        if (out_dims.size()) {
            reduce<src_t, dst_t>(src_data, src_dims, srcStrides, dst_data, dst_dims, dstStrides, 0, keep_dims, skip_dims,
                [](dst_t x, src_t y)->dst_t { return x + y * y; });
        } else {
            dst_data[0] = 0;
            for (src_idx = 0; src_idx < srcStrides[0] * src_dims[0]; ++src_idx)
                dst_data[0] += src_data[src_idx] * src_data[src_idx];
        }
    }
}

class MKLDNNCPUExtReducesTests : public TestsCommon, public WithParamInterface<reduce_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Reduce_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="_IP_" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="axes_for_reduction" type="Input" precision="I32" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="_REDUCE_TYPE_">
            <data keep_dims="_KEEP_DIMS_" />
            <input>
                <port id="1" precision="_IP_">
                    _IN_
                </port>
                <port id="2" precision="I32">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="_OP_">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="2" to-port="2"/>
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
        REPLACE_WITH_STR(model, "_IP_", p.inType);
        REPLACE_WITH_STR(model, "_OP_", p.inType);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.axes_for_reduction.size());
        REPLACE_WITH_STR(model, "_REDUCE_TYPE_", p.reduce_type);
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

    template <typename T>
    static void fill_data_dbgval(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i + 1;
        }
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            reduce_test_params p = ::testing::WithParamInterface<reduce_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            // Input Data
            InferenceEngine::Blob::Ptr src;
            InferenceEngine::SizeVector out_dims;

            InferenceEngine::BlobMap srcs;

            InferenceEngine::Blob::Ptr seq_lengthsIdx;
            InferenceEngine::SizeVector seq_lengths_dim(1, p.axes_for_reduction.size());
            seq_lengthsIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
            seq_lengthsIdx->allocate();
            if (p.axes_for_reduction.size())
                memcpy(static_cast<int32_t*>(seq_lengthsIdx->buffer()), &p.axes_for_reduction[0], sizeof(int32_t)*p.axes_for_reduction.size());
            auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(seq_lengthsIdx.get());
            if (seq_lengthsIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("axes_for_reduction", seq_lengthsIdx));
            if (p.inType == "FP32") {
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();

                src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, p.in_shape,
                                                                InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape)});
                src->allocate();
                if (p.input_tensor.size())
                    for (int i = 0; i < p.input_tensor.size(); i++) {
                        static_cast<float*>(src->buffer())[i] = static_cast<float>(p.input_tensor[i]);
                    }
                else
                    fill_data_dbgval<float>(src->buffer(), src->size());
                auto *srcPtr = dynamic_cast<InferenceEngine::TBlob<float> *>(src.get());
                if (srcPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                ref_reduce<float, float>(p.reduce_type, *srcPtr, p.keep_dims, p.axes_for_reduction, dst_ref, out_dims);
                if (p.reference.size())
                    if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                        FAIL() << "Wrong result with compare reference vector!";
                // Infer
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));
                graph.Infer(srcs, outputBlobs);
                compare(*output, dst_ref);
            } else if (p.inType == "I32") {
                InferenceEngine::TBlob<int32_t>::Ptr output;
                output = InferenceEngine::make_shared_blob<int32_t>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                InferenceEngine::TBlob<int32_t> dst_ref({ InferenceEngine::Precision::I32, p.out_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.out_shape) });
                dst_ref.allocate();

                src = InferenceEngine::make_shared_blob<int32_t>({InferenceEngine::Precision::I32, p.in_shape,
                                                                  InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape)});
                src->allocate();
                if (p.input_tensor.size())
                    for (int i = 0; i < p.input_tensor.size(); i++) {
                        static_cast<int32_t*>(src->buffer())[i] = static_cast<int32_t>(p.input_tensor[i]);
                    }
                else
                    fill_data_dbgval<int32_t>(src->buffer(), src->size());
                auto *srcPtr = dynamic_cast<InferenceEngine::TBlob<int32_t> *>(src.get());
                if (srcPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                ref_reduce<int32_t, int32_t>(p.reduce_type, *srcPtr, p.keep_dims, p.axes_for_reduction, dst_ref, out_dims);
                if (p.reference.size()) {
                    for (int i = 0; i < p.reference.size(); i++) {
                        if (dst_ref.data()[i] != p.reference[i])
                            FAIL() << "Wrong result with compare reference vector!";
                        //std::cout << p.reference[i] << " " << dst_ref.data()[i] << std::endl;
                    }
                }

                // Infer
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));
                graph.Infer(srcs, outputBlobs);
                compare(*output, dst_ref);
            }
            // Check results
            if (out_dims.size() != p.out_shape.size())
                FAIL() << "Wrong out_shape size!";
            for (size_t i = 0; i < p.out_shape.size(); i++) {
                if (out_dims[i] != p.out_shape[i])
                    FAIL() << "Wrong out_shape dimensions!";
            }

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtReducesTests, TestsReduceSum) {}

INSTANTIATE_TEST_CASE_P(
    TestsReduceSum, MKLDNNCPUExtReducesTests,
    ::testing::Values(
        // Params: reduce_type, keep_dims, in_shape, inType, input_tensor, axes_for_reduction, out_shape, reference
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 0 },{ 1, 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ -3 },{ 1, 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 2 },{ 2, 3, 1 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ -1 },{ 2, 3, 1 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 0, 2 },{ 1, 3, 1 },{ 68, 100, 132 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 1, 2 },{ 2, 1, 1 },{ 78, 222 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 2, 1 },{ 2, 1, 1 },{ 78, 222 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 0, 1, 2 },{ 1, 1, 1 },{ 300 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 0, -2, 2 },{ 1, 1, 1 },{ 300 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 2, 2, 2, 2, 2, 2 },"FP32",{},{ 0, 1, 2, 3, 4, 5, 6 },{ 1, 1, 1, 1, 1, 1, 1 },{ 8256 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 2, 2, 2, 2, 2, 2 },"FP32",{},{ 6, 3, 1, 4, 0 },{ 1, 1, 2, 1, 1, 2, 1 },{ 1776, 1840, 2288, 2352 } },
        reduce_test_params{ "ReduceSum", true,{ 2, 3, 4 },"FP32",{},{ 2, 2, 0, 2, 0 },{ 1, 3, 1 },{ 68, 100, 132 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 0 },{ 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ -3 },{ 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 2 },{ 2, 3 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ -1 },{ 2, 3 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 0, 2 },{ 3 },{ 68, 100, 132 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 1, 2 },{ 2 },{ 78, 222 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 2, 1 },{ 2 },{ 78, 222 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 0, 1, 2 },{},{ 300 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 0, -2, 2 },{},{ 300 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 2, 2, 2, 2, 2, 2 },"FP32",{},{ 0, 1, 2, 3, 4, 5, 6 },{},{ 8256 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"FP32",{},{ 2, 2, 0, 2, 0 },{ 3 },{ 68, 100, 132 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 2, 2, 2, 2, 2, 2 },"FP32",{},{ 6, 3, 1, 4, 0 },{ 2, 2 },{ 1776, 1840, 2288, 2352 } },
        reduce_test_params{ "ReduceSum", true,{ 1, 2, 3, 4, 1 },"FP32",{},{ 1 },{ 1, 1, 3, 4, 1 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "ReduceSum", false,{ 1, 2, 3, 4, 1 },"FP32",{},{ 1 },{ 1, 3, 4, 1 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
// I32 tests
        reduce_test_params{ "ReduceAnd", true,{ 2, 2, 2 },"I32",{1, 0, 1, 1, 0, 1, 1, 0},{ 2 },{ 2, 2, 1 },{ 0, 1, 0, 0} },
        reduce_test_params{ "ReduceL1", true, { 3, 2, 2 },"I32",{},{ 2 },{ 3, 2, 1 },{ 3, 7, 11, 15, 19, 23 } },
        reduce_test_params{ "ReduceL1", false, { 3, 2, 2 },"I32",{},{ 0, 1, 2 },{ },{ 78 } },
        reduce_test_params{ "ReduceL2", false,{ 3, 2, 2 },"I32",{},{ 2 },{ 3, 2 },{ 2, 5, 7, 10, 13, 16 } },
        reduce_test_params{ "ReduceL2", false,{ 3, 2, 2 },"I32",{},{ 0, 1, 2 },{ },{ 25 } },
        reduce_test_params{ "ReduceLogSum", true,{ 10, 10, 2 },"I32",{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "ReduceLogSumExp", true,{ 5, 5, 2 },"I32",{},{ 2 },{ 5, 5, 1 },{} },
        reduce_test_params{ "ReduceMax", true,{ 3, 2, 2 },"I32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 20, 2, 40, 2, 60, 2 } },
        reduce_test_params{ "ReduceMean", true, { 3, 2, 2 },"I32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 12, 1, 35, 1, 57, 1 } },
        reduce_test_params{ "ReduceMin", false,{ 3, 2, 2 },"I32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 5, 1, 30, 1, 55, 1 } },
        reduce_test_params{ "ReduceOr", true,{ 2, 2, 2 },"I32",{1, 0, 1, 1, 0, 0, 1, 0},{ 2 },{ 2, 2, 1 },{1, 1, 0, 1 } },
        reduce_test_params{ "ReduceProd", true,{ 3, 2, 2 },"I32",{},{ 1 },{ 3, 1, 2 },{ 3, 8, 35, 48, 99, 120 } },
        reduce_test_params{ "ReduceSum", false,{ 2, 3, 4 },"I32",{},{ 2, 2, 0, 2, 0 },{ 3 },{ 68, 100, 132 } },
        reduce_test_params{ "ReduceSumSquare", true, { 3, 2, 2 },"I32",{},{ 1 },{ 3, 1, 2 },{ 10, 20, 74, 100, 202, 244 } },
        reduce_test_params{ "ReduceSumSquare", false, { 3, 2, 2 },"I32",{},{ 0, 1, 2 },{ },{ 650 } }
));


TEST_P(MKLDNNCPUExtReducesTests, TestsReduceAll) {}

INSTANTIATE_TEST_CASE_P(
    TestsReduceAll, MKLDNNCPUExtReducesTests,
            ::testing::Values(
// Params: reduce_type, keep_dims, in_shape, inType, input_tensor, axes_for_reduction, out_shape, reference
                reduce_test_params{ "ReduceAnd", true,{ 2, 2, 2 },"FP32",{1, 0, 1, 1, 0, 1, 1, 0},{ 2 },{ 2, 2, 1 },{ 0, 1, 0, 0} },
                reduce_test_params{ "ReduceAnd", false, { 2, 2, 2 },"FP32",{1, 0, 1, 1, 0, 1, 1, 0},{ 0, 1, 2 },{ },{ 0 } },
                reduce_test_params{ "ReduceL1", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{ } },
                reduce_test_params{ "ReduceL1", true, { 3, 2, 2 },"FP32",{},{ 2 },{ 3, 2, 1 },{ 3, 7, 11, 15, 19, 23 } },
                reduce_test_params{ "ReduceL1", false, { 3, 2, 2 },"FP32",{},{ 2 },{ 3, 2 },{ 3, 7, 11, 15, 19, 23 } },
                reduce_test_params{ "ReduceL1", false, { 3, 2, 2 },"FP32",{},{ 0, 1, 2 },{ },{ 78 } },
                reduce_test_params{ "ReduceL2", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceL2", true,{ 3, 2, 2 },"FP32",{},{ 2 },{ 3, 2, 1 },{ 2.23606798f, 5.f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f } },
                reduce_test_params{ "ReduceL2", false,{ 3, 2, 2 },"FP32",{},{ 2 },{ 3, 2 },{ 2.23606798f, 5.f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f } },
                reduce_test_params{ "ReduceL2", false,{ 3, 2, 2 },"FP32",{},{ 0, 1, 2 },{ },{ 25.49509757f } },
                reduce_test_params{ "ReduceLogSum", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceLogSum", true,{ 3, 2, 2 },"FP32",{ },{ 1 },{ 3, 1, 2 },{ } },
                reduce_test_params{ "ReduceLogSum", false,{ 3, 2, 2 },"FP32",{ },{ 1 },{ 3, 2 },{ } },
                reduce_test_params{ "ReduceLogSum", false,{ 3, 2, 2 },"FP32",{ },{ 0, 1, 2 },{},{ } },
                reduce_test_params{ "ReduceLogSumExp", true,{ 5, 5, 2 },"FP32",{},{ 2 },{ 5, 5, 1 },{} },
                reduce_test_params{ "ReduceLogSumExp", true,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 20.f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f } },
                reduce_test_params{ "ReduceLogSumExp", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 20.f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f } },
                reduce_test_params{ "ReduceLogSumExp", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 60.00671387f } },
                reduce_test_params{ "ReduceMax", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceMax", true,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 20, 2, 40, 2, 60, 2 } },
                reduce_test_params{ "ReduceMax", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 20, 2, 40, 2, 60, 2 } },
                reduce_test_params{ "ReduceMax", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 60 } },
                reduce_test_params{ "ReduceMean", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceMean", true, { 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 12.5f, 1.5f, 35.f, 1.5f, 57.5f, 1.5f } },
                reduce_test_params{ "ReduceMean", false, { 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 12.5f, 1.5f, 35.f, 1.5f, 57.5f, 1.5f } },
                reduce_test_params{ "ReduceMean", false, { 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{ },{ 18.25f } },
                reduce_test_params{ "ReduceMin", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceMin", true,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 5, 1, 30, 1, 55, 1 } },
                reduce_test_params{ "ReduceMin", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 5, 1, 30, 1, 55, 1 } },
                reduce_test_params{ "ReduceMin", false,{ 3, 2, 2 },"FP32",{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 1 } },
                reduce_test_params{ "ReduceOr", true,{ 2, 2, 2 },"FP32",{1, 0, 1, 1, 0, 0, 1, 0},{ 2 },{ 2, 2, 1 },{1, 1, 0, 1 } },
                reduce_test_params{ "ReduceOr", false, { 2, 2, 2 },"FP32",{},{ 0, 1, 2 },{ },{ 1 } },
                reduce_test_params{ "ReduceProd", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceProd", true,{ 3, 2, 2 },"FP32",{},{ 1 },{ 3, 1, 2 },{ 3, 8, 35, 48, 99, 120 } },
                reduce_test_params{ "ReduceProd", false,{ 3, 2, 2 },"FP32",{},{ 1 },{ 3, 2 },{ 3, 8, 35, 48, 99, 120 } },
                reduce_test_params{ "ReduceProd", false,{ 3, 2, 2 },"FP32",{},{ 0, 1, 2 },{ },{ 4.790016e+08 } },
                reduce_test_params{ "ReduceSumSquare", true,{ 10, 10, 2 },"FP32",{},{ 2 },{ 10, 10, 1 },{} },
                reduce_test_params{ "ReduceSumSquare", true, { 3, 2, 2 },"FP32",{},{ 1 },{ 3, 1, 2 },{ 10, 20, 74, 100, 202, 244 } },
                reduce_test_params{ "ReduceSumSquare", false, { 3, 2, 2 },"FP32",{},{ 1 },{ 3, 2 },{ 10, 20, 74, 100, 202, 244 } },
                reduce_test_params{ "ReduceSumSquare", false, { 3, 2, 2 },"FP32",{},{ 0, 1, 2 },{ },{ 650 } }
));
