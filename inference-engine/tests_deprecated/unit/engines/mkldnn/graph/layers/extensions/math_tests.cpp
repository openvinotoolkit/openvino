// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"
#include "mkldnn_graph.h"
#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include <cpp/ie_cnn_net_reader.h>


using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct math_test_params {
    std::string                 math_function;
    InferenceEngine::SizeVector in_out;
    std::vector<float>          input_tensor;
    std::vector<float>          alpha;
    std::vector<float>          beta;
    std::vector<float>          gamma;
    std::vector<float>          reference;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_math(
    std::string                    math_function,
    InferenceEngine::TBlob<float> &src,
    std::vector<float>             alpha,
    std::vector<float>             beta,
    std::vector<float>             gamma,
    InferenceEngine::TBlob<float> &dst
) {
    size_t i;
    float* src_data = src.data();
    float *dst_data = dst.data();
    size_t dst_size = dst.size();

    if (math_function == "Erf") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = std::erf(src_data[i]);
        }
    } else if (math_function == "Abs") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = (std::abs)(src_data[i]);
        }
    } else if (math_function == "Acos") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = acosf(src_data[i]);
        }
    } else if (math_function == "Acosh") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = acoshf(src_data[i]);
        }
    } else if (math_function == "Asin") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = asinf(src_data[i]);
        }
    } else if (math_function == "Asinh") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = asinhf(src_data[i]);
        }
    } else if (math_function == "Atan") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = atanf(src_data[i]);
        }
    } else if (math_function == "Atanh") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = atanhf(src_data[i]);
        }
    } else if (math_function == "Ceil") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = ceilf(src_data[i]);
        }
    } else if (math_function == "Cos") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = cosf(src_data[i]);
        }
    } else if (math_function == "Cosh") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = coshf(src_data[i]);
        }
    } else if (math_function == "Floor") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = floorf(src_data[i]);
        }
    } else if (math_function == "HardSigmoid") {
        alpha[0] = (alpha[0] == 0.0f) ? 0.2f : alpha[0];
        beta[0] = (beta[0] == 0.0f) ? 0.5f : beta[0];
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = (std::max)(0.f, (std::min)(1.f, alpha[0] * src_data[i] + beta[0]));
        }
    } else if (math_function == "Log") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = logf(src_data[i]);
        }
    } else if (math_function == "Neg") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = -src_data[i];
        }
    } else if (math_function == "Reciprocal") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = 1.0f / src_data[i];
        }
    } else if (math_function == "Selu") {
        alpha[0] = (alpha[0] == 0.0f) ? 1.67326f : alpha[0];
        gamma[0] = (gamma[0] == 0.0f) ? 1.0507f : gamma[0];
        for (i = 0; i < dst_size; i++) {
            float x = src_data[i];
            dst_data[i] = (x > 0.0f) ? (gamma[0] * x) : (gamma[0] * alpha[0] * (exp(x) - 1.0f));
        }
    } else if (math_function == "Sign") {
        for (i = 0; i < dst_size; i++) {
            if (src_data[i] > 0.0f) dst_data[i] = 1.0f;
            else if (src_data[i] < 0.0f) dst_data[i] = -1.0f;
            else dst_data[i] = 0.0f;
        }
    } else if (math_function == "Sin") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = sinf(src_data[i]);
        }
    } else if (math_function == "Sinh") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = sinhf(src_data[i]);
        }
    } else if (math_function == "Softplus") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = logf(expf(src_data[i]) + 1);
        }
    } else if (math_function == "Softsign") {
        for (i = 0; i < dst_size; i++) {
            float x = src_data[i];
            dst_data[i] = x / (1.f + (std::abs)(x));
        }
    } else if (math_function == "Tan") {
        for (i = 0; i < dst_size; i++) {
            dst_data[i] = tanf(src_data[i]);
        }
    }
}

class MKLDNNCPUExtMathTests: public TestsCommon, public WithParamInterface<math_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Math_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="Input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="math" id="2" type="_MATH_FUNCTION_" precision="FP32">
            <data _ALPHA_ _BETA_ _GAMMA_/>
            <input>
                <port id="1">
                    _IN_OUT_
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
    </edges>
</net>
)V0G0N";

    std::string getModel(math_test_params p) {
        std::string model = model_t;
        std::string in_out = "";
        std::string alpha;
        std::string beta;
        std::string gamma;

        for (auto& dst : p.in_out) {
            in_out += "<dim>";
            in_out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IN_OUT_", in_out);
        REPLACE_WITH_STR(model, "_MATH_FUNCTION_", p.math_function);

        if (p.alpha.size()) {
            alpha = "alpha=\"" + to_string_c_locale(p.alpha[0]) + "\"";
        }
        REPLACE_WITH_STR(model, "_ALPHA_", alpha);

        if (p.beta.size()) {
            beta = "beta=\"" + to_string_c_locale(p.beta[0]) + "\"";
        }
        REPLACE_WITH_STR(model, "_BETA_", beta);

        if (p.gamma.size()) {
            gamma = "gamma=\"" + to_string_c_locale(p.gamma[0]) + "\"";
        }
        REPLACE_WITH_STR(model, "_GAMMA_", gamma);
        return model;
    }

    template <typename data_t>
    static void fill_data_dbgval(data_t *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<data_t>(i & (sizeof(data_t) * 8 - 1));
        }
    }
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            math_test_params p = ::testing::WithParamInterface<math_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            // Input Data
            InferenceEngine::Blob::Ptr srcData = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_out, InferenceEngine::TensorDesc::getLayoutByDims(p.in_out) });
            srcData->allocate();
            if (p.input_tensor.size())
                memcpy(srcData->buffer(), &p.input_tensor[0], sizeof(float)*p.input_tensor.size());
            else {
                if (p.math_function == "Erf")
                    CommonTestUtils::fill_data_sine(srcData->buffer(), srcData->size(), 0.f, 3.f, 1.f);
                else
                    CommonTestUtils::fill_data(srcData->buffer(), srcData->size());
            }
            auto * srcDataPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcData.get());
            if (srcDataPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

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

            // Check results
            ref_math(p.math_function, *srcDataPtr, p.alpha, p.beta, p.gamma, dst_ref);
            if (p.reference.size()) {
                for (size_t i = 0; i < p.reference.size(); i++) {
                    ASSERT_NEAR(dst_ref.data()[i], p.reference[i], 0.00001f);
                }
            }

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("Input", srcData));

            // Infer
            graph.Infer(srcs, outputBlobs);
            float threshold = p.math_function == "Erf" ? 0.0001f : 0.00001f;
            compare(*output, dst_ref, threshold);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtMathTests, TestsMath) {}

INSTANTIATE_TEST_CASE_P(
        TestsMath, MKLDNNCPUExtMathTests,
            ::testing::Values(
                // Params: math_function, in_out, input_tensor, alpha, beta, gamma, reference
                math_test_params{ "Erf", {},{},{},{},{},{} },
                math_test_params{ "Erf", { 1, 1, 12, 256 }, {},{},{},{}, {} },
                math_test_params{ "Erf", { 12, 256, 3 },{},{},{},{},{} },
                math_test_params{ "Erf", { 3, 4 },{},{},{},{},{} },
                math_test_params{ "Erf", { 20 },{},{},{},{},{} },
                math_test_params{ "Erf", { 12, 4, 9, 8 },{},{},{},{},{} },
                math_test_params{ "Erf", { 6, 12, 4, 9, 8, 10, 3 },{},{},{},{},{} },
                math_test_params{ "Abs",{ 3 },{ -1, 0, 1 },{},{},{},{ 1, 0, 1 } },
                math_test_params{ "Acos",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ 2.09439516f, 1.57079637f, 1.04719758f } },
                math_test_params{ "Acosh",{ 3 },{ 1.f, 2.0f, 3.0f },{},{},{},{} },
                math_test_params{ "Asin",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ -0.523598790f, 0.0f, 0.523598790f } },
                math_test_params{ "Asinh",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ } },
                math_test_params{ "Atan",{ 3 },{ -1, 0, 1 },{},{},{},{ -0.785398185f, 0.0f, 0.785398185f } },
                math_test_params{ "Atanh",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ } },
                math_test_params{ "Ceil",{ 2 },{ -1.5f, 1.2f },{},{},{},{ -1, 2 } },
                math_test_params{ "Cos",{ 3 },{ -1, 0, 1 },{},{},{},{ 0.540302336f, 1.0f, 0.540302336f } },
                math_test_params{ "Cosh",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ } },
                math_test_params{ "Floor",{ 3 },{-1.5f, 1.2f, 2.f},{},{},{},{-2, 1, 2} },
                math_test_params{ "HardSigmoid",{ 3 },{ -1, 0, 1 },{0.5f},{0.6f},{},{ 0.1f, 0.6f, 1.f } },
                math_test_params{ "Log",{ 2 },{ 1, 10 },{},{},{},{ 0.f, 2.30258512f } },
                math_test_params{ "Neg",{ 3 },{ -1, 0, 1 },{},{},{},{ 1, 0, -1 } },
                math_test_params{ "Reciprocal",{ 3 },{ -1, 0.1, 1 },{2},{},{3},{-1, 10, 1} },
                math_test_params{ "Selu",{ 3 },{ -1, 0, 1 },{2},{},{3},{ -3.79272318f, 0.f, 3.f } },
                math_test_params{ "Sign",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{-1, 0, 1} },
                math_test_params{ "Sin",{ 3 },{ -1, 0, 1 },{},{},{},{ -0.841470957f, 0.0f, 0.841470957f } },
                math_test_params{ "Sinh",{ 3 },{ -0.5f, 0.f, 0.5f },{},{},{},{ } },
                math_test_params{ "Softplus",{ 3 },{ -1, 0, 1 },{},{},{},{ 0.31326166f, 0.69314718f, 1.31326163f } },
                math_test_params{ "Softsign",{ 3 },{ -1, 0, 1 },{},{},{},{ -0.5f, 0.f, 0.5f } },
                math_test_params{ "Tan",{ 3 },{ -1, 0, 1 },{},{},{},{ -1.55740774f, 0.0f, 1.55740774f } }
            ));
