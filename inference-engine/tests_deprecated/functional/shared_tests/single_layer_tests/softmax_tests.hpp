// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;


struct softmax_base_params {
    struct {
        size_t w;
        size_t h;
        size_t c;
        size_t n;
    } in;

    int axis;
};

struct softmax_test_params : softmax_base_params {
    std::string device_name;
    std::string model;

    softmax_test_params(std::string name, softmax_base_params params, std::string model = "4D") :
            softmax_base_params(params), device_name(name), model(model) {}
};

template <typename data_t>
void check_softmax_fwd(const data_t *src_data, softmax_test_params prm)
{
  size_t W = prm.in.w;
  size_t H = prm.in.h;
  size_t C = prm.in.c;
  size_t MB = prm.in.n;

  auto off = [=](int n, int c, int h, int w)
  {
    return (n * W * H * C + c * W * H + h * W + w);
  };

  double result = 0.0f;

  if(prm.axis == 0) {

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          result = 0.0f;
          for (int n = 0; n < MB; ++n) {
            result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
          }

          ASSERT_NEAR(result, 1.0f, 0.001);
        }
      }
    }
  }
  else if(prm.axis == 1) {
    for (int n = 0; n < MB; ++n) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          result = 0.0f;

          for (int c = 0; c < C; ++c) {
            result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
          }

          ASSERT_NEAR(result, 1.0f, 0.001);
        }
      }
    }
  }
  else if(prm.axis == 2) {
    for (int n = 0; n < MB; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int w = 0; w < W; ++w) {
          result = 0.0f;

          for (int h = 0; h < H; ++h) {
            result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
          }

          ASSERT_NEAR(result, 1.0f, 0.001);
        }
      }
    }
  }
  else if(prm.axis == 3) {
    for (int n = 0; n < MB; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          result = 0.0f;

          for (int w = 0; w < W; ++w) {
            result += src_data[off(n, c, h, w)];//dst_ptr[map_index(dst_pd, off(n, c, h, w))];
          }

          ASSERT_NEAR(result, 1.0f, 0.001);
        }
      }
    }
  }
}

class SoftmaxOnlyTest: public TestsCommon,
                    public WithParamInterface<softmax_test_params> {

    std::string model_t = R"V0G0N(
    <Net Name="SoftmaxOnly" version="2" precision="FP32" batch="_IB_">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="softmax" id="1" type="Softmax" precision="FP32">
            <input>
                <port id="0">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</Net>
)V0G0N";

    std::string model_2D = R"V0G0N(
    <Net Name="SoftmaxOnly" version="2" precision="FP32" batch="_IB_">
    <layers>
        <layer name="input_1" type="input" id="0" precision="FP32">
            <output>
                <port id="0">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                </port>
            </output>
        </layer>
        <layer name="softmax" id="1" type="Softmax" precision="FP32">
            <input>
                <port id="0">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IB_</dim>
                    <dim>_IC_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    </edges>
</Net>
)V0G0N";

    std::string getModel(softmax_test_params p) {
        std::string model = p.model == "2D" ? model_2D :  model_t;
        REPLACE_WITH_NUM(model, "_IB_", p.in.n);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        return model;
    }

protected:
    virtual void SetUp() {

        try {
            softmax_test_params p = ::testing::WithParamInterface<softmax_test_params>::GetParam();
            std::string model = getModel(p);

            bool is2D = p.model == "2D";

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());

            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            if (p.in.n != 1) {
                net.setBatchSize(p.in.n);
            }

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();
            
            auto src = inferRequest.GetBlob(in_info_map.begin()->first);
            auto src_data = src->buffer().as<float*>();
            for (int i=0; i != p.in.n; i++) {
                fill_data(src_data + p.in.w * p.in.h * p.in.c * i, src->size() / p.in.n);
            }

            inferRequest.Infer();
            auto dst = inferRequest.GetBlob(out_info_map.begin()->first);

            check_softmax_fwd(dst->buffer().as<float*>(), p);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1 softmax_base_params({{228, 228, 3, 1}, 1})
#define case_8 softmax_base_params({{228, 228, 3, 8}, 1})
#define case_8_nc softmax_base_params({{1, 1, 228*228*3, 8}, 1})

TEST_P(SoftmaxOnlyTest, TestsSoftmax) {}

std::string  getTestCaseName(testing::TestParamInfo<softmax_test_params> obj) {
    return obj.param.device_name +
           "_h" + std::to_string(obj.param.in.h) +
           "_w" + std::to_string(obj.param.in.w) +
           "_c" + std::to_string(obj.param.in.c) +
           "_b" + std::to_string(obj.param.in.n);
}
