// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include "defs.h"
#include "softmax.h"
#include <vector>
#include "simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class RegionYoloImpl: public ExtLayerBase {
public:
    explicit RegionYoloImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            classes = layer->GetParamAsInt("classes");
            coords = layer->GetParamAsInt("coords");
            num = layer->GetParamAsInt("num");
            do_softmax = static_cast<bool>(layer->GetParamAsInt("do_softmax", 1));
            mask = layer->GetParamAsInts("mask", {});

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int mask_size = mask.size();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        simple_copy(dst_data, outputs[0]->byteSize(), src_data, (size_t)B * IC * IH * IW * sizeof(float));

        int end_index = 0;
        int num_ = 0;
        if (do_softmax) {
            // Region layer (Yolo v2)
            end_index = IW * IH;
            num_ = num;
        } else {
            // Yolo layer (Yolo v3)
            end_index = IW * IH * (classes + 1);
            num_ = mask_size;
        }
        int inputs_size = IH * IW * num_ * (classes + coords + 1);

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < num_; n++) {
                int index = entry_index(IW, IH, coords, classes, inputs_size, b, n * IW * IH, 0);
                for (int i = index; i < index + 2 * IW * IH; i++) {
                    dst_data[i] = logistic_activate(dst_data[i]);
                }

                index = entry_index(IW, IH, coords, classes, inputs_size, b, n * IW * IH, coords);
                for (int i = index; i < index + end_index; i++) {
                    dst_data[i] = logistic_activate(dst_data[i]);
                }
            }
        }

        if (do_softmax) {
            int index = entry_index(IW, IH, coords, classes, inputs_size, 0, 0, coords + 1);
            int batch_offset = inputs_size / num;
            for (int b = 0; b < B * num; b++)
                softmax_generic(src_data + index + b * batch_offset, dst_data + index + b * batch_offset, 1, classes,
                                IH, IW);
        }

        return OK;
    }

private:
    int classes;
    int coords;
    int num;
    float do_softmax;
    std::vector<int> mask;

    inline int entry_index(int width, int height, int coords, int classes, int outputs, int batch, int location,
                           int entry) {
        int n = location / (width * height);
        int loc = location % (width * height);
        return batch * outputs + n * width * height * (coords + classes + 1) +
               entry * width * height + loc;
    }

    inline float logistic_activate(float x) {
        return 1.f / (1.f + exp(-x));
    }
};

REG_FACTORY_FOR(ImplFactory<RegionYoloImpl>, RegionYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
