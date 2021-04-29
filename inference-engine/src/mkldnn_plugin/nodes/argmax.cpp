// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include "argmax_imp.hpp"

#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ArgMaxImpl: public ExtLayerBase {
public:
    explicit ArgMaxImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                IE_THROW() << "Incorrect number of input/output edges!";

            conf.out_max_val_ = layer->GetParamAsBool("out_max_val", false);
            conf.top_k_       = layer->GetParamAsInt("top_k");

            conf.has_axis_ = (layer->params.find("axis") != layer->params.end());
            conf.axis_index_ = conf.has_axis_ ?
                                std::stoi(layer->params.at("axis")) :0;

            addConfig(layer, {DataConfigurator(ConfLayout::PLN, Precision::FP32)}, {DataConfigurator(ConfLayout::PLN, Precision::FP32)});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        SizeVector in_dims = inputs[0]->getTensorDesc().getDims();

        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        XARCH::arg_max_execute(src_data, dst_data, in_dims, conf);
        return OK;
    }

private:
    argmax_conf conf;
};

REG_FACTORY_FOR(ArgMaxImpl, ArgMax);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
