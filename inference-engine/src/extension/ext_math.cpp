// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class MathImpl: public ExtLayerBase {
    static float error_function(float x) {
        const float clip_bound = 2.86f;
        //  Points clip_bound and -clip_bound are extremums for this polynom
        //  So in order to provide better accuracy comparing to std::erf we have to clip input range
        if (x > clip_bound)
            return 1;
        if (x < -clip_bound)
            return -1;

        //  A polynomial approximation of the error function
        const float erfNumerator[4] = { 90.0260162353515625f, 2232.00537109375f,
            7003.3251953125f, 55592.30078125f };
        const float erfDenominator[5] = { 33.56171417236328125f, 521.35797119140625f,
            4594.32373046875f, 22629.0f, 49267.39453125f };
        float polynom = 9.60497379302978515625f;
        float x2 = x * x;
        for (float c : erfNumerator) {
            polynom = polynom * x2 + c;
        }
        x *= polynom;
        polynom = 1.0f;
        for (float c : erfDenominator) {
            polynom = polynom * x2 + c;
        }
        return x / polynom;
    }

public:
    explicit MathImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            if (layer->insData[0].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32 is supported!";

            if (layer->insData[0].lock()->getTensorDesc().getDims() != layer->outData[0]->getTensorDesc().getDims())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            alpha = layer->GetParamAsFloat("alpha", 0.0f);
            beta = layer->GetParamAsFloat("beta", 0.0f);
            gamma = layer->GetParamAsFloat("gamma", 0.0f);

            std::string math_func = layer->type;
            if (math_func == "Erf") mathFunction = Math::Erf;
            else if (math_func == "Abs") mathFunction = Math::Abs;
            else if (math_func == "Acos") mathFunction = Math::Acos;
            else if (math_func == "Acosh") mathFunction = Math::Acosh;
            else if (math_func == "Asin") mathFunction = Math::Asin;
            else if (math_func == "Asinh") mathFunction = Math::Asinh;
            else if (math_func == "Atan") mathFunction = Math::Atan;
            else if (math_func == "Atanh") mathFunction = Math::Atanh;
            else if (math_func == "Ceil") mathFunction = Math::Ceil;
            else if (math_func == "Cos") mathFunction = Math::Cos;
            else if (math_func == "Cosh") mathFunction = Math::Cosh;
            else if (math_func == "Floor") mathFunction = Math::Floor;
            else if (math_func == "HardSigmoid") mathFunction = Math::HardSigmoid;
            else if (math_func == "Log") mathFunction = Math::Log;
            else if (math_func == "Neg") mathFunction = Math::Neg;
            else if (math_func == "Reciprocal") mathFunction = Math::Reciprocal;
            else if (math_func == "Selu") mathFunction = Math::Selu;
            else if (math_func == "Sign") mathFunction = Math::Sign;
            else if (math_func == "Sin") mathFunction = Math::Sin;
            else if (math_func == "Sinh") mathFunction = Math::Sinh;
            else if (math_func == "Softplus") mathFunction = Math::Softplus;
            else if (math_func == "Softsign") mathFunction = Math::Softsign;
            else if (math_func == "Tan") mathFunction = Math::Tan;
            else
                THROW_IE_EXCEPTION << layer->name << " Incorrect Math layer type!";

            addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        size_t dataSize = outputs[0]->size();
        const float *src_data = inputs[0]->cbuffer().as<const float *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        switch (mathFunction) {
        case Math::Erf:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = error_function(src_data[i]);
            });
            break;
        case Math::Abs:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::abs)(src_data[i]);
            });
            break;
        case Math::Acos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acosf(src_data[i]);
            });
            break;
        case Math::Acosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acoshf(src_data[i]);
            });
            break;
        case Math::Asin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinf(src_data[i]);
            });
            break;
        case Math::Asinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinhf(src_data[i]);
            });
            break;
        case Math::Atan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanf(src_data[i]);
            });
            break;
        case Math::Atanh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanhf(src_data[i]);
            });
            break;
        case Math::Ceil:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = ceilf(src_data[i]);
            });
            break;
        case Math::Cos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = cosf(src_data[i]);
            });
            break;
        case Math::Cosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = coshf(src_data[i]);
            });
            break;
        case Math::Floor:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = floorf(src_data[i]);
            });
            break;
        case Math::HardSigmoid:
            alpha = (alpha == 0.0f) ? 0.2f : alpha;
            beta = (beta == 0.0f) ? 0.5f : beta;
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::max)(0.f, (std::min)(1.f, alpha * src_data[i] + beta));
            });
            break;
        case Math::Log:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(src_data[i]);
            });
            break;
        case Math::Neg:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = -src_data[i];
            });
            break;
        case Math::Reciprocal:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = 1.0f / src_data[i];
            });
            break;
        case Math::Selu:
            alpha = (alpha == 0.0f) ? 1.67326f : alpha;
            gamma = (gamma == 0.0f) ? 1.0507f : gamma;
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = (x > 0.0f) ? (gamma * x) : (gamma * alpha * (exp(x) - 1.0f));
            });
            break;
        case Math::Sign:
            parallel_for(dataSize, [&](size_t i) {
                if (src_data[i] > 0.0f)
                    dst_data[i] = 1.0f;
                else if (src_data[i] < 0.0f)
                    dst_data[i] = -1.0f;
                else
                    dst_data[i] = 0.0f;
            });
            break;
        case Math::Sin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinf(src_data[i]);
            });
            break;
        case Math::Sinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinhf(src_data[i]);
            });
            break;
        case Math::Softplus:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(expf(src_data[i]) + 1);
            });
            break;
        case Math::Softsign:
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = x / (1.f + (std::abs)(x));
            });
            break;
        case Math::Tan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = tanf(src_data[i]);
            });
            break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect Reduce layer type";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    enum class Math {
        Abs,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Ceil,
        Cos,
        Cosh,
        Erf,
        Floor,
        HardSigmoid,
        Log,
        Neg,
        Reciprocal,
        Selu,
        Sign,
        Sin,
        Sinh,
        Softplus,
        Softsign,
        Tan
    };

    Math mathFunction = Math::Erf;
    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;
};

REG_FACTORY_FOR(ImplFactory<MathImpl>, Abs);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Acos);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Acosh);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Asin);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Asinh);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Atan);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Atanh);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Ceil);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Cos);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Cosh);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Erf);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Floor);
REG_FACTORY_FOR(ImplFactory<MathImpl>, HardSigmoid);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Log);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Neg);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Reciprocal);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Selu);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Sign);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Sin);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Sinh);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Softplus);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Softsign);
REG_FACTORY_FOR(ImplFactory<MathImpl>, Tan);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
