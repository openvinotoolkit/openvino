// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"


using namespace ::testing;
using namespace InferenceEngine;
using namespace std;


struct reverse_sequence_test_params {
    std::string device_name;
    std::string inPrecision;
    SizeVector in_out_shape;
    std::vector<int32_t> seq_lengths;
    int seq_axis;
    int batch_axis;
    std::vector<float> reference;
};

template <typename data_t>
void ref_reverse_sequence(
        TBlob<float> &src,
        TBlob<data_t> &seq_lengths,
        TBlob<float> &dst,
        int seq_axis,
        int batch_axis
) {
    size_t i, src_idx;
    const float *src_data = src.data();
    SizeVector src_dims = src.getTensorDesc().getDims();
    SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    const data_t *seq_lengths_data = seq_lengths.data();
    SizeVector seq_lengths_dims = seq_lengths.getTensorDesc().getDims();
    float* dst_data = dst.data();

    if (seq_axis < 0)
        seq_axis += src_dims.size();

    if (seq_axis < 0 || seq_axis >= src_dims.size())
        FAIL() << "Incorrect 'seq_axis' parameters dimensions and axis number!";

    if (batch_axis < 0)
        batch_axis += src_dims.size();

    if (batch_axis < 0 || batch_axis >= src_dims.size())
        FAIL() << "Incorrect 'batch_axis' parameters dimensions and axis number!";

    for (i = 0; i < src_dims[batch_axis]; i++) {
        if (static_cast<int32_t>(seq_lengths_data[i]) > src_dims[seq_axis])
            FAIL() << "Incorrect input 'seq_lengths' values!";
    }

    size_t work_amount_dst = srcStrides[0] * src_dims[0];
    SizeVector counters(src_dims.size(), 0);
    for (size_t iwork = 0; iwork < work_amount_dst; ++iwork) {
        for (i = 0, src_idx = 0; i < src_dims.size(); ++i) {
            size_t idx = counters[i];
            if (i == seq_axis && idx < static_cast<int32_t>(seq_lengths_data[counters[batch_axis]])) {
                idx = static_cast<int32_t>(seq_lengths_data[counters[batch_axis]]) - idx - 1;
            }
            src_idx += idx * srcStrides[i];
        }

        dst_data[iwork] = src_data[src_idx];

        for (int j = src_dims.size() - 1; j >= 0; j--) {
            counters[j] = (counters[j] + 1) % src_dims[j];
            if (counters[j] != 0) break;
        }
    }
}

class ReverseSequenceTests : public TestsCommon, public WithParamInterface<reverse_sequence_test_params> {
    std::string model_t = R"V0G0N(
<net Name="ReverseSequence_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="_INP_" id="1">
            <output>
                <port id="1">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="seq_lengths" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="ReverseSequence" id="2" type="ReverseSequence" precision="FP32">
            <data seq_axis="_SA_" batch_axis="_BA_"/>
            <input>
                <port id="1">
                    _IN_OUT_
                </port>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
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
        <edge from-layer="2" from-port="2" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(reverse_sequence_test_params p) {
        std::string model = model_t;
        std::string in_out_shape;
        for (size_t i = 0; i < p.in_out_shape.size(); i++) {
            in_out_shape += "<dim>";
            in_out_shape += std::to_string(p.in_out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_INP_", p.inPrecision);
        REPLACE_WITH_STR(model, "_IN_OUT_", in_out_shape);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.seq_lengths.size());
        REPLACE_WITH_NUM(model, "_SA_", p.seq_axis);
        REPLACE_WITH_NUM(model, "_BA_", p.batch_axis);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            reverse_sequence_test_params p = ::testing::WithParamInterface<reverse_sequence_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            // Output Data
            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();

            // Output Reference
            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Input Data
            auto src = make_shared_blob<float>({ Precision::FP32,
                p.in_out_shape,
                TensorDesc::getLayoutByDims(p.in_out_shape) });
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());

            SizeVector seq_lengths_dim(1, p.seq_lengths.size());
            auto seq_lengthsIdx = make_shared_blob<float>({ Precision::FP32,
                seq_lengths_dim,
                TensorDesc::getLayoutByDims(seq_lengths_dim) });
            seq_lengthsIdx->allocate();
            if (p.seq_lengths.size())
                for (size_t i = 0; i < p.seq_lengths.size(); i++) {
                    static_cast<float *>(seq_lengthsIdx->buffer())[i] = static_cast<float>(p.seq_lengths[i]);
                }

            auto * seq_lengthsIdxPtr = dynamic_cast<TBlob<float>*>(seq_lengthsIdx.get());
            if (seq_lengthsIdxPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            ref_reverse_sequence(*srcPtr, *seq_lengthsIdxPtr, dst_ref, p.seq_axis, p.batch_axis);
            if (p.reference.size()) {
                if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                    FAIL() << "Wrong result with compare TF reference!";
            }

            ExecutableNetwork executable_network = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            inferRequest.SetBlob("input", src);
            inferRequest.SetBlob("seq_lengths", seq_lengthsIdx);

            inferRequest.SetBlob(item.first, output);
            inferRequest.Infer();

            // Check results
            compare(*output, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

//  Test data vectors
static std::vector<float> test1 = { 3.f,4.f,5.f,0.f,1.f,2.f,6.f,7.f,8.f,12.f,13.f,14.f,9.f,10.f,11.f,15.f,16.f,17.f,21.f,22.f,23.f,18.f,19.f,20.f,24.f,25.f,26.f };
static std::vector<float> test2 = { 1.f,0.f,2.f,4.f,3.f,5.f,7.f,6.f,8.f,10.f,9.f,11.f,13.f,12.f,14.f,16.f,15.f,17.f,19.f,18.f,20.f,22.f,21.f,23.f,25.f,24.f,26.f };
static std::vector<float> test3 = { 2.f,1.f,0.f,4.f,3.f,5.f };
static std::vector<float> test4 = { 0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,12.f,13.f,14.f,9.f,10.f,11.f,15.f,16.f,17.f,24.f,25.f,26.f,21.f,22.f,23.f,18.f,19.f,20.f };
static std::vector<float> test5 = { 0.f,4.f,8.f,3.f,1.f,5.f,6.f,7.f,2.f,9.f,13.f,17.f,12.f,10.f,14.f,15.f,16.f,11.f,18.f,22.f,26.f,21.f,19.f,23.f,24.f,25.f,20.f };
static std::vector<float> test6 = { 0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,13.f,12.f,15.f,14.f,17.f,16.f,19.f,18.f,21.f,20.f,23.f,22.f };

TEST_P(ReverseSequenceTests, smoke_GPU_TestsReverseSequence) {}
INSTANTIATE_TEST_CASE_P(
        smoke_TestsReverseSequence, ReverseSequenceTests,
        ::testing::Values(
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 2, 2, 2 },  1, 0, test1 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 2, 2, 2 }, -2, 0, test1 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 2, 2, 2 },  2, 1, test2 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 2, 2, 2 }, -1, 1, test2 },
        reverse_sequence_test_params{"GPU", "FP32", { 2, 3 },{ 3, 2 }, 1, 0, test3 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 1, 2, 3 },  1, 0, test4 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 1, 2, 3 },  1,-3, test4 },
        reverse_sequence_test_params{"GPU", "FP32", { 3, 3, 3 },{ 1, 2, 3 },  1, 2, test5 },
        reverse_sequence_test_params{"GPU", "FP32", { 2, 2, 3, 2 },{ 1, 2 }, 3, 0, test6 }
));
