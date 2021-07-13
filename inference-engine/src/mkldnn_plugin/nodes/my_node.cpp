// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include <string>
#include <vector>
#include <ngraph/op/my_node.hpp>
#include "common/tensor_desc_creator.h"
#include "utils/general_utils.h"

namespace InferenceEngine {
    namespace Extensions {
        namespace Cpu {
            using MKLDNNPlugin::TensorDescCreatorTypes;

            class MyNodeImpl: public ExtLayerBase {
            public:
                bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
                    return true;
                }

                explicit MyNodeImpl(const std::shared_ptr<ngraph::Node>& op) {
                    try {
                        if (!isSupportedOperation(op, errorMsg)) {
                            IE_THROW(NotImplemented) << errorMsg;
                        }
                        auto myOp = ngraph::as_type_ptr<ngraph::op::v0::MyNode>(op);

                        SizeVector dstDims = myOp->get_output_shape(0);
                        SizeVector srcDims = myOp->get_input_shape(0);
                        if (srcDims.size() != dstDims.size()) {
                            IE_THROW(ParameterMismatch) << errorMsg;
                        }
                        for (size_t i = 0; i < srcDims.size(); i++) {
                            if (srcDims[i] != dstDims[i]) {
                                IE_THROW(ParameterMismatch) << errorMsg;
                            }
                        }
                        addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                                  {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
                       } catch (InferenceEngine::Exception &ex) {
                        errorMsg = ex.what();
                        throw;
                    }
                }

                StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
                    const float *src = inputs[0]->cbuffer().as<float *>() +
                                       inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                    float *dst = outputs[0]->buffer().as<float *>() +
                               outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                    size_t size = inputs[0]->size();
                    for (size_t i = 0; i < size; i++) {
                        dst[i] = src[i];
                    }
                    return OK;
                }
            };
            REG_FACTORY_FOR(MyNodeImpl, MyNode);
        }  // namespace Cpu
    }  // namespace Extensions
}  // namespace InferenceEngine
