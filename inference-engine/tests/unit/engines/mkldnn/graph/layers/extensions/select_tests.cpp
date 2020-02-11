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


using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace InferenceEngine;

using select_test_params = std::tuple<
    InferenceEngine::Precision,     // conditionType
    InferenceEngine::SizeVector,    // conditionShape
    InferenceEngine::SizeVector     // inputShape
>;

template<typename T>
void ref_select(
    InferenceEngine::TBlob<T>     &condition,
    InferenceEngine::TBlob<float> &then_,
    InferenceEngine::TBlob<float> &else_,
    InferenceEngine::TBlob<float> &dst
) {
    const T *conditionData = condition.buffer();

    const float *thenData = then_.cbuffer().as<const float *>();

    const float *elseData = else_.cbuffer().as<const float *>();

    float* dstData = dst.cbuffer().as<float *>();
    enum {N, C, H, W, Dims};
    int dim[Dims] = {1, 1, 1, 1};
    int cdim[Dims] = {1, 1, 1, 1};

    InferenceEngine::SizeVector dims = then_.getTensorDesc().getDims();
    std::copy(std::begin(dims), std::end(dims), std::begin(dim) + (Dims - dims.size()));

    InferenceEngine::SizeVector cDims = condition.getTensorDesc().getDims();
    std::copy(std::begin(cDims), std::end(cDims), std::begin(cdim) + (Dims - cDims.size()));

    for (int b = 0; b < dim[N]; b++)
    for (int c = 0; c < dim[C]; c++)
    for (int h = 0; h < dim[H]; h++)
    for (int w = 0; w < dim[W]; w++) {
                dstData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
        = conditionData[(b % cdim[N])*cdim[C]*cdim[H]*cdim[W] + (c % cdim[C])*cdim[H]*cdim[W] + (h % cdim[H])*cdim[W] + (w % cdim[W])]
        ?      thenData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
        :      elseData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w];
    }
}

class MKLDNNCPUExtSelectTests : public TestsCommon, public WithParamInterface<select_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Select_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="condition" type="Input" precision="_CONDITION_TYPE_" id="0">
            <output>
                <port id="0">
                    _CONDITION_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="then_" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="else_" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
        <layer name="select" type="Select" precision="FP32" id="3">
            <input>
                <port id="0">
                    _CONDITION_SHAPE_
                </port>
                <port id="1">
                    _INPUT_SHAPE_
                </port>
                <port id="2">
                    _INPUT_SHAPE_
                </port>
            </input>
            <output>
                <port id="3">
                    _INPUT_SHAPE_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(InferenceEngine::Precision  conditionType,
                         InferenceEngine::SizeVector conditionShape,
                         InferenceEngine::SizeVector inputShape) {
        std::string model = model_t;

        {
            std::string conditionTypeStr;
            switch(conditionType) {
                case InferenceEngine::Precision::FP32 : conditionTypeStr = "FP32"; break;
                case InferenceEngine::Precision::I32  : conditionTypeStr = "I32" ; break;
                default: EXPECT_FALSE("Unsuported pressision");
            }
            REPLACE_WITH_STR(model, "_CONDITION_TYPE_", conditionTypeStr);
        }

        {
            std::string conditionShapeStr;
            for (auto dim : conditionShape) {
                conditionShapeStr += "<dim>";
                conditionShapeStr += std::to_string(dim) + "</dim>\n";
            }
            conditionShapeStr.pop_back();
            REPLACE_WITH_STR(model, "_CONDITION_SHAPE_", conditionShapeStr);
        }

        {
            std::string inputShapeStr;
            for (auto dim : inputShape) {
                inputShapeStr += "<dim>";
                inputShapeStr += std::to_string(dim) + "</dim>\n";
            }
            inputShapeStr.pop_back();
            REPLACE_WITH_STR(model, "_INPUT_SHAPE_", inputShapeStr);
        }

        return model;
    }

    static void fill_even(int32_t *data, size_t size) {
        for (size_t i = 0; i < size; i++)
            data[i] = i%2 ? 1 : 0;
    }


protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            Precision  conditionType;
            SizeVector conditionShape;
            SizeVector inputShape;
            std::tie(conditionType, conditionShape, inputShape) = ::testing::WithParamInterface<select_test_params>::GetParam();
            std::string model = getModel(conditionType, conditionShape, inputShape);
            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

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

            // Input Data
            InferenceEngine::Blob::Ptr then_;
            then_ = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                                                               inputShape, InferenceEngine::TensorDesc::getLayoutByDims(inputShape) });
            then_->allocate();
            fill_data_dbgval(then_->buffer(), then_->size());
            auto * thenPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(then_.get());
            if (thenPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Input Data
            InferenceEngine::Blob::Ptr else_;
            else_ = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32,
                                                               inputShape, InferenceEngine::TensorDesc::getLayoutByDims(inputShape) });
            else_->allocate();
            fill_data_dbgval(else_->buffer(), else_->size(), -1.0);
            auto * elsePtr = dynamic_cast<InferenceEngine::TBlob<float>*>(else_.get());
            if (elsePtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::Blob::Ptr condition;

            switch (conditionType) {
            case Precision::FP32 :
                condition = make_shared_blob<float>({ conditionType, conditionShape,
                                                  TensorDesc::getLayoutByDims(conditionShape) });
                condition->allocate();
                fill_data(condition->buffer(), condition->size());

                break;
            case Precision::I32 :
                condition = make_shared_blob<int32_t>({ conditionType, conditionShape,
                                                        TensorDesc::getLayoutByDims(conditionShape) });
                break;
            default:
                FAIL();
                break;
            }

            condition->allocate();
            fill_even(condition->buffer(), condition->size());

            switch (conditionType) {
            case InferenceEngine::Precision::FP32 : {
                auto conditionPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<float>>(condition);
                if (conditionPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
                ref_select(*conditionPtr, *thenPtr, *elsePtr, dst_ref);
            }
            break;
            case InferenceEngine::Precision::I32 : {
                auto conditionPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int32_t>>(condition);
                if (conditionPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";
                ref_select(*conditionPtr, *thenPtr, *elsePtr, dst_ref);
            }
            break;
            default:
                FAIL();
            }

            InferenceEngine::BlobMap srcs = {
                {"condition", condition},
                {"then_",     then_},
                {"else_",     else_},
            };

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtSelectTests, TestsSelect) {}
INSTANTIATE_TEST_CASE_P(
    TestsSelect, MKLDNNCPUExtSelectTests,
            Combine(
                Values(Precision::I32),
                Values(
//                       SizeVector {},  // TODO: scalars is not supported right now for CPU backend
                       SizeVector {1},
                       SizeVector {1, 1},
                       SizeVector {1, 16},
                       SizeVector {3, 1, 16},
                       SizeVector {1, 16, 1},
                       SizeVector {3, 16, 16}),
                Values(SizeVector {3, 16, 16})
            ));
