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


struct permute_base_params {
    SizeVector dims;
    SizeVector order;
};

struct permute_test_params {
    std::string device_name;
    permute_base_params base;
    permute_test_params(std::string name, permute_base_params params) : device_name(name), base(params) {}
};

template <typename data_t>
void ref_permute(const TBlob<data_t> &src, TBlob<data_t> &dst, permute_base_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    SizeVector orderedDims;
    for (auto ord : prm.order) {
        orderedDims.push_back(src.getTensorDesc().getDims()[ord]);
    }
    TensorDesc desc(Precision::FP32, src.getTensorDesc().getDims(), {orderedDims, prm.order});

    for (int i=0; i < src.size(); i++) {
        dst_data[desc.offset(i)] = src_data[src.getTensorDesc().offset(i)];
    }
}

class PermuteOnlyTests: public TestsCommon,
                        public WithParamInterface<permute_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Power_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    __DIMS__
                </port>
            </output>
        </layer>
        <layer name="permute" id="1" type="Permute" precision="FP32">
            <data order="_ORDER_"/>
            <input>
                <port id="1">
                    __DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</Net>
)V0G0N";

protected:
    std::string getModel(permute_base_params p) {
        std::string model = model_t;
        std::string dims;
        std::string dst_dims;
        for (auto& dim : p.dims) {
            dims += "<dim>";
            dims += std::to_string(dim) + "</dim>\n";
        }

        std::string order;
        for (auto& ord : p.order) {
            if (!order.empty())
                order += ",";
            order += std::to_string(ord);
            dst_dims += "<dim>";
            dst_dims += std::to_string(p.dims[ord]) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "__DIMS__", dims);
        REPLACE_WITH_STR(model, "__DST_DIMS__", dst_dims);
        REPLACE_WITH_STR(model, "_ORDER_", order);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            permute_test_params p = ::testing::WithParamInterface<permute_test_params>::GetParam();
            std::string model = getModel(p.base);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            Blob::Ptr src = make_shared_blob<float>({Precision::FP32, p.base.dims,
                                        TensorDesc::getLayoutByDims(p.base.dims)});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto* srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            inferRequest.SetBlob("in1", src);

            OutputsDataMap out = net.getOutputsInfo();
            auto item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);
            inferRequest.Infer();

            TensorDesc td(Precision::FP32, p.base.dims,
                                           TensorDesc::getLayoutByDims(p.base.dims));
            TBlob<float> dst_ref(td);
            dst_ref.allocate();

            ref_permute(*srcPtr, dst_ref, p.base);

            compare(*output, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(PermuteOnlyTests, TestsPermute) {}

#define case_1  permute_base_params{{2, 3, 4, 5}, {0, 1, 2, 3}}
#define case_2  permute_base_params{{2, 3, 4, 5}, {0, 2, 3, 1}}
#define case_3  permute_base_params{{2, 3, 4, 5}, {0, 2, 1, 3}}
#define case_4  permute_base_params{{2, 3, 4}, {0, 1, 2}}
#define case_5  permute_base_params{{2, 3, 4}, {0, 2, 1}}
#define case_6  permute_base_params{{2, 3}, {0, 1}}
#define case_7  permute_base_params{{2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}}
#define case_8  permute_base_params{{2, 3, 4, 5, 6}, {0, 4, 2, 1, 3}}
#define case_9  permute_base_params{{2, 3, 4, 5, 6}, {0, 2, 4, 3, 1}}
#define case_10 permute_base_params{{2, 3, 4, 5, 6}, {0, 3, 2, 4, 1}}
#define case_11 permute_base_params{{2, 8, 2, 2, 4, 5}, {0, 1, 4, 2, 5, 3}}
#define case_12 permute_base_params{{2, 8, 3, 3, 4, 5}, {0, 1, 4, 2, 5, 3}}
#define case_13 permute_base_params{{2, 12, 9}, {0, 2, 1}}
#define case_14 permute_base_params{{2, 8, 3, 3, 4, 5}, {0, 3, 4, 1, 5, 2}}
#define case_15 permute_base_params{{2, 3, 4, 5}, {0, 1, 3, 2}}
#define case_16 permute_base_params{{2, 3, 4, 5, 7}, {0, 3, 1, 4, 2}}
