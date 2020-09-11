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


struct strided_slice_test_params {
    std::string device_name;
    InferenceEngine::SizeVector in_dim;
    std::vector<int> begin;
    std::vector<int> end;
    std::vector<int> strides;
    InferenceEngine::SizeVector ref_dim;
    std::vector<float> ref;
};

inline void clipping(int *idx, const int min, const int max) {
    (*idx) = ((*idx) > min) ? (*idx) : min;
    (*idx) = ((*idx) < max) ? (*idx) : (max - 1);
    return;
}

void ref_strided_slice(
        InferenceEngine::TBlob<float> &src,
        InferenceEngine::TBlob<float> &dst,
        InferenceEngine::SizeVector &out_dims,
        std::vector<int> begin,
        std::vector<int> end,
        std::vector<int> stride,
        InferenceEngine::SizeVector begin_mask,
        InferenceEngine::SizeVector end_mask,
        InferenceEngine::SizeVector ellipsis_mask,
        InferenceEngine::SizeVector new_axis_mask,
        InferenceEngine::SizeVector shrink_axis_mask
) {
    size_t i;
    const float *src_data = src.data();
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    float* dst_data = dst.data();
    InferenceEngine::SizeVector dst_dims = dst.getTensorDesc().getDims();
    InferenceEngine::SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();

    int new_axis = 0;
    for (auto& na : new_axis_mask)
        new_axis += na;

    int shrink_axis = 0;
    for (auto& sa : shrink_axis_mask)
        shrink_axis += sa;
    int max_dims = src_dims.size() + new_axis;

    //  Check beging/end/stride vector sizes
    int bounds_size = 0;
    if (begin.size() && end.size() && begin.size() != end.size()) FAIL() << "Begin vector size should be equal end vectror size";
    if (begin.size() && stride.size() && stride.size() != begin.size()) FAIL() << "Stride vector size should be equal begin vectror size";
    if (end.size() && stride.size() && stride.size() != end.size()) FAIL() << "Stride vector size should be equal end vectror size";

    if (begin.size()) bounds_size = begin.size();
    if (end.size()) bounds_size = end.size();
    if (stride.size()) bounds_size = stride.size();

    //  ellipsis_mask must be a power of two (only one ellipsis), so to take a first position
    int ellipsis_pos1, ellipsis_pos2;
    ellipsis_pos1 = ellipsis_pos2 = max_dims;
    for (i = 0; i < ellipsis_mask.size(); i++) {
        if (ellipsis_mask[i] > 0) {
            ellipsis_pos1 = i;
            break;
        }
    }
    bounds_size -= ellipsis_pos1;
    if(bounds_size > 0 && (max_dims - bounds_size) > ellipsis_pos1)
        ellipsis_pos2 = max_dims - bounds_size;

    std::vector<int> begin_dms(max_dims, 0);
    std::vector<int> end_dms(max_dims, -1);
    std::vector<int> stride_dms(max_dims, 1);

    int j, k, bj, ej, sj;
    InferenceEngine::SizeVector our_dims;
    for (i = 0, j = 0, k = 0, bj = 0, ej = 0, sj = 0; i < max_dims; i++) {
        if (i >= ellipsis_pos1 && i < ellipsis_pos2) {
            if (!(new_axis_mask.size() > i && new_axis_mask[i] == 1)) {
                end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j++] + end_dms[i];
            } else {
                //end_dms[i] = 0;
                end_dms[i] = begin_dms[i];
            }
            out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));
            our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));
            k = ellipsis_pos1;
            continue;
        }
        stride_dms[i] = (stride.size() > sj && stride[sj] != 0) ? stride[sj++] : 1;

        if (!(begin_mask.size() > j && begin_mask[j] == 0))
            begin_dms[i] = begin.size() > bj ? begin[bj] : (stride_dms[i] > 0 ? 0 : -1);
        else
            begin_dms[i] = stride_dms[i] > 0 ? 0 : -1;
        bj++;
        begin_dms[i] = begin_dms[i] >= 0 ? begin_dms[i] : src_dims[j] + begin_dms[i];
        //  Clipping 'begin'
        clipping(&begin_dms[i], 0, src_dims[j]);

        if (!(end_mask.size() > j && end_mask[j] == 0)) {
            int end_dms_tmp = end.size() > ej ? (stride_dms[i] > 0 ? end[ej] - 1 : end[ej] + 1) : end_dms[i];
            end_dms[i] = end.size() > ej ? end_dms_tmp : (stride_dms[i] > 0 ? -1 : 0);
        }
        else {
            end_dms[i] = stride_dms[i] > 0 ? -1 : 0;
        }
        ej++;
        end_dms[i] = end_dms[i] >= 0 ? end_dms[i] : src_dims[j] + end_dms[i];
        //  Clipping 'end'
        clipping(&end_dms[i], 0, src_dims[j]);

        if (!(new_axis_mask.size() > i && new_axis_mask[i] == 1))
            j++;
        else
            end_dms[i] = 0;

        if (shrink_axis_mask.size() > k && shrink_axis_mask[k] == 1)
            end_dms[i] = begin_dms[i];
        else
            out_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));

        our_dims.push_back(static_cast<int>(ceil(static_cast<float>(abs(end_dms[i] - begin_dms[i]) + 1) / static_cast<float>(abs(stride_dms[i])))));
        k++;
    }

    size_t work_amount_dst = dstStrides[0] * dst_dims[0];
    InferenceEngine::SizeVector counters(max_dims, 0);

    for (size_t iwork = 0, dst_idx = 0; iwork < work_amount_dst; ++iwork) {
        int src_idx = 0;
        for (i = 0, j = 0; i < max_dims; ++i) {
            src_idx += (begin_dms[i] + counters[i] * stride_dms[i]) * srcStrides[j];
            if (!(new_axis_mask.size() > i && new_axis_mask[i] == 1)) j++;
        }

        dst_data[dst_idx++] = src_data[src_idx];

        for (j = max_dims - 1; j >= 0; j--) {
            counters[j] = (counters[j] + 1) % our_dims[j];
            if (counters[j] != 0) break;
        }
    }
}

template<typename data_t>
void ref_strided_slice(std::vector<Blob::Ptr> &dsts, strided_slice_test_params& prm) {
    data_t *dst_data = dsts[0]->buffer().as<data_t*>();

    for(int i = 0; i < prm.ref.size(); ++i)
        dst_data[i] = prm.ref[i];
}

InferenceEngine::TBlob<uint8_t>::Ptr generateWeights(const std::vector<std::vector<int>> &data) {
    size_t totalSize = 0;
    for (size_t i = 0; i < data.size(); ++i)
        totalSize += data[i].size();
    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(
        { InferenceEngine::Precision::U8,{ totalSize * sizeof(uint32_t) }, Layout::C }
        );
    weights->allocate();
    size_t vectorCounter = 0;
    size_t innerVectorCounter = 0;
    for (size_t i = 0; i < totalSize; i++) {
        if (innerVectorCounter >= data[vectorCounter].size()) {
            ++vectorCounter;
            innerVectorCounter = 0;
        }
        ((uint32_t*) weights->buffer())[i] = data[vectorCounter][innerVectorCounter];
        ++innerVectorCounter;
    }
    return InferenceEngine::TBlob<uint8_t>::Ptr(weights);
}

class StridedSliceTests : public TestsCommon, public WithParamInterface<strided_slice_test_params> {
    std::string model_t = R"V0G0N(
<net Name="strided_slice" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="Input1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer id="2" name="Input2" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="4"/>
            </blobs>
        </layer>
        <layer id="3" name="Input3" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="16" size="4"/>
            </blobs>
        </layer>
        <layer id="4" name="Input4" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="32" size="4"/>
            </blobs>
        </layer>
        <layer name="strided_slice" id="5" type="StridedSlice" precision="FP32">
            <data begin_mask=""
                  end_mask=""
                  ellipsis_mask=""
                  new_axis_mask=""
                  shrink_axis_mask=""/>
            <input>
                <port id="5">
                    _IN_
                </port>
                <port id="6">
                    <dim>4</dim>
                </port>
                <port id="7">
                    <dim>4</dim>
                </port>
                <port id="8">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="9">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="5"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="6"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="7"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="8"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(strided_slice_test_params p) {
        std::string in, out;

        for (auto& i : p.in_dim) {
            in += "<dim>" + std::to_string(i) + "</dim>\n";
        }

        for (auto& o : p.ref_dim) {
            out += "<dim>" + std::to_string(o) + "</dim>\n";
        }

        REPLACE_WITH_STR(model_t, "_IN_", in);
        REPLACE_WITH_STR(model_t, "_OUT_", out);

        return model_t;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            strided_slice_test_params p = ::testing::WithParamInterface<strided_slice_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, generateWeights({ p.begin, p.end, p.strides }));
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            InferenceEngine::OutputsDataMap out;
            out = net.getOutputsInfo();

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Input Data
            InferenceEngine::Blob::Ptr src;
            src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_dim, InferenceEngine::TensorDesc::getLayoutByDims(p.in_dim) });
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            ref_strided_slice(*srcPtr, dst_ref, p.ref_dim, p.begin, p.end, p.strides, {}, {}, {}, {}, {});

            inferRequest.SetBlob("Input1", src);

            inferRequest.Infer();

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(StridedSliceTests, smoke_GPU_TestsStridedSlice) {}

//  Test data vectors
std::vector<float> ref1 = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };
std::vector<float> ref2 = { 15.f };
std::vector<float> ref3 = { 0.f, 1.f, 2.f, 6.f, 7.f, 8.f, 12.f, 13.f, 14.f, 18.f, 19.f, 20.f, 24.f, 25.f, 26.f, 30.f, 31.f, 32.f, 36.f, 37.f, 38.f, 42.f, 43.f, 44.f };
std::vector<float> ref4 = { 33.f, 34.f, 35.f, 41.f, 42.f, 43.f, 49.f, 50.f, 51.f, 57.f, 58.f, 59.f };
std::vector<float> ref5 = { 0.f, 1.f, 2.f, 8.f, 9.f, 10.f, 12.f, 13.f, 14.f, 20.f, 21.f, 22.f, 24.f, 25.f, 26.f, 32.f, 33.f, 34.f, 36.f, 37.f, 38.f, 44.f, 45.f, 46.f };

INSTANTIATE_TEST_CASE_P(
        smoke_TestsStridedSlice, StridedSliceTests,
        ::testing::Values(
                strided_slice_test_params{ "GPU", { 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, ref1 },
                strided_slice_test_params{ "GPU", { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, ref2 },
                strided_slice_test_params{ "GPU", { 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 }, { 2, 2, 2, 3 }, ref3 },
                strided_slice_test_params{ "GPU", { 2, 2, 4, 4 }, { 1, 0, 0, 1 }, { 2, 2, 4, 4 }, { 1, 1, 2, 1 }, { 1, 2, 2, 3 }, ref4 },
                strided_slice_test_params{ "GPU", { 2, 2, 3, 4 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 }, { 2, 2, 2, 3 }, ref5 }
        ));
