// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ie_core.hpp"
#include "../common_single_layer_tests/pool_ref.hpp"
#include "common_test_utils/common_layers_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct pooling_base_params {
    struct { size_t n, c, h, w; } in;
    struct { size_t h, w; } out;

    size_t krn_h;
    size_t krn_w;
    size_t str_h;
    size_t str_w;
    size_t pad_h;
    size_t pad_w;

    bool avg;
    bool exclude_pad;
};

struct pooling_test_params : pooling_base_params {
    std::string device_name;

    pooling_test_params(std::string name, pooling_base_params params) :
            pooling_base_params(params), device_name(name) {}
};

template <typename data_t>
void ref_pool(const Blob::Ptr &src, Blob::Ptr &dst, pooling_test_params p)
{
    CommonTestUtils::pool_common_params params;
    params.kernel.insert(X_AXIS, p.krn_w);
    params.kernel.insert(Y_AXIS, p.krn_h);
    params.stride.insert(X_AXIS, p.str_w);
    params.stride.insert(Y_AXIS, p.str_h);
    params.pads_begin.insert(X_AXIS, p.pad_w);
    params.pads_begin.insert(Y_AXIS, p.pad_h);
    params.exclude_pad = p.exclude_pad;
    params.avg = p.avg;
    ref_pool_common<float>({ src }, *dst.get(), params);
}

class smoke_CPU_PoolingOnlyTest: public TestsCommon,
                       public WithParamInterface<pooling_test_params> {

    std::string model_t = R"V0G0N(
<net name="Pooling_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="pool" type="Pooling" precision="FP32">

            <data
                exclude-pad="_EXCL_PAD_"
                pool-method="_ALG_"
                kernel-x="_KW_" kernel-y="_KH_"
                pad-x="_PW_" pad-y="_PH_"
                stride-x="_SW_" stride-y="_SH_"  />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(pooling_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IN_", p.in.n);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);

        REPLACE_WITH_NUM(model, "_KH_", p.krn_h);
        REPLACE_WITH_NUM(model, "_KW_", p.krn_w);
        REPLACE_WITH_NUM(model, "_SH_", p.str_h);
        REPLACE_WITH_NUM(model, "_SW_", p.str_w);
        REPLACE_WITH_NUM(model, "_PH_", p.pad_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pad_w);

        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        REPLACE_WITH_STR(model, "_ALG_", p.avg ? "avg":"max");
        REPLACE_WITH_STR(model, "_EXCL_PAD_", p.exclude_pad ? "true":"false");

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            pooling_test_params p = ::testing::WithParamInterface<pooling_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());

            SizeVector dims_src = {p.in.w, p.in.h, p.in.c, p.in.n};
            Blob::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), NCHW));
            src->allocate();
            fill_data(src->buffer().as<float *>(), src->size());

            SizeVector dims_dst = {p.out.w, p.out.h, p.in.c, p.in.n};
            Blob::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst->allocate();

            Blob::Ptr dst_ref = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), NCHW));
            dst_ref->allocate();

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.device_name);
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            OutputsDataMap outInfo;
            outInfo = network.getOutputsInfo();
            ASSERT_EQ(outInfo.size(), 1);
            ASSERT_NE(outInfo.begin()->second, nullptr);
            inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            inferRequest.SetBlob(outInfo.begin()->first, dst);
            inferRequest.Infer();

            ref_pool<float>(src, dst_ref, p);
            compare(*dst.get(), *dst_ref.get());

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_0 pooling_base_params({{1, 3, 228, 228}, {114, 114}, 2, 2, 2, 2, 0, 0})
#define case_1 pooling_base_params({{1, 3, 228, 228}, {113, 114}, 4, 2, 2, 2, 0, 0})
#define case_2 pooling_base_params({{1, 3, 228, 228}, {113, 227}, 4, 2, 2, 1, 0, 0})
#define case_3 pooling_base_params({{1, 3, 224, 224}, {224, 224}, 3, 3, 1, 1, 1, 1, false, false})
#define case_4 pooling_base_params({{1, 3, 224, 224}, {224, 224}, 3, 3, 1, 1, 1, 1, true, false})
#define case_5 pooling_base_params({{1, 3, 224, 224}, {224, 224}, 3, 3, 1, 1, 1, 1, true, true})

#define case_6 pooling_base_params({{1, 3, 224, 224}, {112, 112}, 3, 3, 2, 2, 1, 1, false, false})
#define case_7 pooling_base_params({{1, 3, 224, 224}, {112, 112}, 3, 3, 2, 2, 1, 1, true, false})
#define case_8 pooling_base_params({{1, 3, 224, 224}, {112, 112}, 3, 3, 2, 2, 1, 1, true, true})

#define case_9  pooling_base_params({{1, 3, 224, 224}, {113, 113}, 3, 3, 2, 2, 1, 1, false, false})
#define case_10 pooling_base_params({{1, 3, 224, 224}, {113, 113}, 3, 3, 2, 2, 1, 1, true, false})
#define case_11 pooling_base_params({{1, 3, 224, 224}, {113, 113}, 3, 3, 2, 2, 1, 1, true, true})


TEST_P(smoke_CPU_PoolingOnlyTest, TestsPooling) {}

std::string  getTestCaseName(testing::TestParamInfo<pooling_test_params> obj) {
    return  obj.param.device_name +
        "_w" + std::to_string(obj.param.in.w) +
        "_h" + std::to_string(obj.param.in.h) +
        "_c" + std::to_string(obj.param.in.c) +
        "_krnw" + std::to_string(obj.param.krn_w) +
        "_krnh" + std::to_string(obj.param.krn_h) +
        "_strw" + std::to_string(obj.param.str_w) +
        "_strh" + std::to_string(obj.param.str_h);
}

pooling_test_params pooling_only_test_cases[] = {
        pooling_test_params("CPU", case_0),
        pooling_test_params("CPU", case_1),
		pooling_test_params("CPU", case_2),
		pooling_test_params("CPU", case_3),
        pooling_test_params("CPU", case_4),
        pooling_test_params("CPU", case_5),
        pooling_test_params("CPU", case_6),
        pooling_test_params("CPU", case_7),
        pooling_test_params("CPU", case_8),
        pooling_test_params("CPU", case_9),
        pooling_test_params("CPU", case_10),
        pooling_test_params("CPU", case_11),
};

INSTANTIATE_TEST_CASE_P(
        TestsPooling, smoke_CPU_PoolingOnlyTest, ::testing::ValuesIn(pooling_only_test_cases));
