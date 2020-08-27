// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_add_const_infer.hpp"
#include "ie_div_const_infer.hpp"
#include "ie_mul_const_infer.hpp"
#include "ie_pow_const_infer.hpp"
#include "ie_sub_const_infer.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 * @brief Eltwise wrapper on top of Mul/Add/Div operation
 */
class EltwiseConstInfer : public ConstInferImpl {
public:
    explicit EltwiseConstInfer(const std::string& type): ConstInferImpl(type) {
        _sum = std::shared_ptr<ConstInferImpl>(new AddConstInfer(_type));
        _sub = std::shared_ptr<ConstInferImpl>(new SubConstInfer(_type));
        _mul = std::shared_ptr<ConstInferImpl>(new MulConstInfer(_type));
        _div = std::shared_ptr<ConstInferImpl>(new DivConstInfer(_type));
        _pow = std::shared_ptr<ConstInferImpl>(new PowConstInfer(_type));
    }

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        auto found = params.find("operation");
        IE_ASSERT(found != params.end()) << "Eltwise layer has no attribute operation.";
        std::string operation = found->second;

        std::shared_ptr<ConstInferImpl> actual;
        if (operation == "sum")
            actual = _sum;
        else if (operation == "sub")
            actual = _sub;
        else if (operation == "mul")
            actual = _mul;
        else if (operation == "div")
            actual = _div;
        else if (operation == "pow")
            actual = _pow;
        else
            THROW_IE_EXCEPTION << "Unsupported eltwise operation type " << operation
                               << ". "
                                  "IE cannot propagate constants through this layer.";

        actual->inferImpl(inData, params, blobs, outData);
    }

private:
    std::shared_ptr<ConstInferImpl> _mul, _div, _sum, _sub, _pow;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
