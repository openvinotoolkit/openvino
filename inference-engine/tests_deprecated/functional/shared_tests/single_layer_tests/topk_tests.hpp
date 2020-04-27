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

struct topk_test_params {
    std::string          device_name;
    SizeVector           in_shape;
    int                  axis;
    std::vector<size_t>  src_k;
    std::string          sort;
    std::string          mode;
    SizeVector           out_shape;
    Precision            precision;
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

template <typename T>
static void ref_topk(TBlob<T> &src, TBlob<T> &dst_data, TBlob<int> &dst_indx, topk_test_params p) {
    T* src_data = src.data();
    T* dst_val = dst_data.data();
    int* dst_idx = dst_indx.data();

    int dim, axis_dist;
    int src_k = static_cast<int>(p.src_k[0]);


    SizeVector src_dims = src.getTensorDesc().getDims();;
    int axis_ = p.axis;
    if (axis_ < 0)
        axis_ += src_dims.size();

    size_t axis = static_cast<size_t>(axis_);

    if (src_dims.size() < (1 + axis))
        FAIL() << " Incorrect input parameters dimensions and axis number!";

    bool mode_max;
    if (p.mode == "max")
        mode_max = true;
    else
        mode_max = false;

    bool sort_value;
    if (p.sort == "value")
        sort_value = true;
    else
        sort_value = false;

    int j;
    for (j = src_dims.size() - 1; j >= 0; j--) {
        if (src_dims[j] != 1) break;
    }
    if (static_cast<size_t>(j) == axis) {
        dim = count(src_dims, static_cast<size_t>(j));
        axis_dist = 1;
    } else {
        int axis_ = (p.axis < 0) ? p.axis + static_cast<int>(src_dims.size()) : p.axis;
        dim = static_cast<int>(src_dims[axis_]);
        axis_dist = count(src_dims, axis_) / dim;
    }

    int num = count(src_dims) / dim;
    std::vector<std::pair<T, int> > src_vector(src_k);

    for (int i = 0; i < num; ++i) {
        src_vector[0] = std::make_pair(src_data[(i / axis_dist * dim) * axis_dist + i % axis_dist], 0);
        for (j = 1; j < src_k; ++j) {
            src_vector[j] = std::make_pair(src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
            if (mode_max) {
                if (src_vector[j].first > src_vector[j - 1].first)
                    std::sort(src_vector.begin(), src_vector.begin() + j + 1, std::greater<std::pair<T, int> >());
            } else {
                if (src_vector[j].first < src_vector[0].first)
                    std::sort(src_vector.begin(), src_vector.begin() + j + 1, std::less<std::pair<T, int> >());
            }
        }

        for (; j < dim; ++j) {
            T value = src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist];
            if (mode_max) {
                if (value > src_vector[src_k - 1].first) {
                    src_vector[src_k - 1] = std::make_pair(value, j);
                    std::sort(src_vector.begin(), src_vector.end(), std::greater<std::pair<T, int> >());
                }
            } else {
                if (value < src_vector[0].first) {
                    src_vector[src_k - 1] = std::make_pair(value, j);
                    std::sort(src_vector.begin(), src_vector.end(), std::less<std::pair<T, int> >());
                }
            }
        }

        if (!sort_value)
            std::sort(src_vector.begin(), src_vector.begin() + src_k, [&src_vector](const pair<int, int> &a, const pair<int, int> &b)
            { return (a.second < b.second); });

        for (int j = 0; j < src_k; ++j) {
            if (axis_dist != 1) {
                // Produces max_val per axis
                dst_val[(i / axis_dist * src_k + j) * axis_dist + i % axis_dist] = src_vector[j].first;
                dst_idx[(i / axis_dist * src_k + j) * axis_dist + i % axis_dist] = src_vector[j].second;
            } else {
                // Produces max_ind and max_val
                dst_val[i * src_k + j] = src_vector[j].first;
                dst_idx[i * src_k + j] = src_vector[j].second;
            }
        }
    }
}

template <typename src_data_t>
class TopKTests : public TestsCommon, public WithParamInterface<topk_test_params> {
    std::string model_t = (std::string)R"V0G0N(
<net Name="TopK_net" version="2" precision="_SRC_DATA_T_" batch="1">
    <layers>
        <layer name="value" type="Input" precision="_SRC_DATA_T_" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="src_k" type="Const" precision="I32" id="2">
            <output>
                <port id="2"/>
            </output>
            <blobs>
                <custom offset="0" size="1"/>
            </blobs>
        </layer>
        <layer name="output" id="3" type="TopK">
            <data axis="_AXIS_" sort="_SORT_" mode="_MODE_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2"/>
            </input>
            <output>
                <port id="3" precision="_SRC_DATA_T_">
                    _OUT_
                </port>
                <port id="4" precision="I32">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(topk_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        for (auto& dct : p.in_shape) {
            in_shape += "<dim>";
            in_shape += std::to_string(dct) + "</dim>\n";
        }

        switch (p.precision) {
            case Precision::FP32:
                REPLACE_WITH_STR(model, "_SRC_DATA_T_", "FP32"); break;
            case Precision::I32:
                REPLACE_WITH_STR(model, "_SRC_DATA_T_", "I32"); break;
            default:
                THROW_IE_EXCEPTION << "Unsupported test precision";
        }

        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_SORT_", p.sort);
        REPLACE_WITH_STR(model, "_MODE_", p.mode);
        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            topk_test_params p = ::testing::WithParamInterface<topk_test_params>::GetParam();
            std::string model = getModel(p);


            TBlob<uint8_t>* top_k = new TBlob<uint8_t>(
                { Precision::U8,{ p.src_k.size() * sizeof(int32_t) }, Layout::C });
            top_k->allocate();
            for (size_t i = 0; i < p.src_k.size(); i++) {
                ((int32_t *) top_k->buffer())[i] = p.src_k[i];
            }
            
            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, TBlob<uint8_t>::Ptr(top_k));
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Output Data
            OutputsDataMap out;
            out = net.getOutputsInfo();
            BlobMap outputBlobs;

            auto it = out.begin();
            std::pair<std::string, DataPtr> item0 = *it;
            std::pair<std::string, DataPtr> item1 = *(++it);

            typename TBlob<src_data_t>::Ptr output0;
            output0 = make_shared_blob<src_data_t>(item0.second->getTensorDesc());
            output0->allocate();
            inferRequest.SetBlob(item0.first, output0);
            TBlob<int>::Ptr output1;
            output1 = make_shared_blob<int>(item1.second->getTensorDesc());
            output1->allocate();
            inferRequest.SetBlob(item1.first, output1);


            // Input Data
            Blob::Ptr src;
            src = make_shared_blob<src_data_t>({ p.precision, p.in_shape, TensorDesc::getLayoutByDims(p.in_shape) });
            src->allocate();
            for (size_t i = 0; i < src->size(); i++) {
                src->buffer().as<src_data_t*>()[i] = i % 2 == 0 ? static_cast<src_data_t>(i) : static_cast<src_data_t>(-1.f * i - i * 2);
            }

            inferRequest.SetBlob("value", src);

            // Output Reference
            TBlob<src_data_t> dst_data_ref(item0.second->getTensorDesc());
            dst_data_ref.allocate();
            TBlob<int> dst_indx_ref(item1.second->getTensorDesc());
            dst_indx_ref.allocate();
            auto* srcPtr = dynamic_cast<TBlob<src_data_t>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<src_data_t>.";
            ref_topk<src_data_t>(*srcPtr, dst_data_ref, dst_indx_ref, p);

            inferRequest.Infer();

            for (size_t i = 0; i < dst_data_ref.size(); i++) {
                if (dst_data_ref.buffer().template as<src_data_t*>()[i] != output0.get()->buffer().template as<src_data_t*>()[i]) {
                    FAIL() << "The difference between ref_val " << dst_data_ref.buffer().template as<src_data_t*>()[i] <<
                              " and res_val " << output0.get()->buffer().template as<src_data_t*>()[i] << " at " << i << " index";
                }
            }

            for (size_t i = 0; i < dst_data_ref.size(); i++) {
                if (dst_indx_ref.buffer().as<int*>()[i] != output1.get()->buffer().as<int*>()[i]) {
                    FAIL() << "The difference between ref_idx " << dst_indx_ref.buffer().as<int*>()[i] <<
                           " and res_idx " << output1.get()->buffer().as<int*>()[i] << " at " << i << " index";
                }
            }
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

using topk_test_int32 = TopKTests<int32_t>;
using topk_test_fp32 = TopKTests<float>;

TEST_P(topk_test_int32, TestsTopK_I32) {}

TEST_P(topk_test_fp32, TestsTopK_FP32) {}

