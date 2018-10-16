// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include <functional>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ArgMaxImpl: public ExtLayerBase {
public:
    explicit ArgMaxImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            out_max_val_ = static_cast<bool>(layer->GetParamAsInt("out_max_val"));
            top_k_       = layer->GetParamAsInt("top_k");

            has_axis_ = (layer->params.find("axis") != layer->params.end());
            axis_index_ = has_axis_ ?
                                std::stoi(layer->params.at("axis")) :0;

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        SizeVector in_dims = inputs[0]->getTensorDesc().getDims();
        SizeVector out_dims = outputs[0]->getTensorDesc().getDims();

        int dim, axis_dist;
        if (has_axis_) {
            int axis_ = (axis_index_ < 0) ? axis_index_ + static_cast<int>(in_dims.size()) : axis_index_;
            dim = static_cast<int>(inputs[0]->getTensorDesc().getDims()[axis_]);
            axis_dist = count(inputs[0]->getTensorDesc().getDims(), axis_) / dim;
        } else {
            dim = count(inputs[0]->getTensorDesc().getDims(), 1);
            axis_dist = 1;
        }

        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        int num = count(in_dims) / dim;
        std::vector<std::pair<float, int> > src_vector(dim);

        for (int i = 0; i < num; ++i) {
            for (int j = 0; j < dim; ++j) {
                src_vector[j] = std::make_pair(
                        src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
            }

            std::partial_sort(src_vector.begin(), src_vector.begin() + top_k_,
                              src_vector.end(), std::greater<std::pair<float, int> >());

            for (int j = 0; j < top_k_; ++j) {
                if (out_max_val_) {
                    if (has_axis_) {
                        // Produces max_val per axis
                        dst_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist] = src_vector[j].first;
                    } else {
                        // Produces max_ind and max_val
                        dst_data[2 * i * top_k_ + j] = src_vector[j].second;
                        dst_data[2 * i * top_k_ + top_k_ + j] = src_vector[j].first;
                    }
                } else {
                    // Produces max_ind per axis
                    dst_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist] = src_vector[j].second;
                }
            }
        }

        return OK;
    }

private:
    bool out_max_val_;
    int top_k_;
    bool has_axis_;
    int axis_index_;

    inline int count(SizeVector dims, size_t start_ind, size_t end_ind) {
        size_t count = 1;
        for (size_t i = start_ind; i < end_ind; i++)
            count *= dims[i];
        return static_cast<int>(count);
    }

    inline int count(SizeVector dims, size_t start_ind = 0) {
        return count(dims, start_ind, dims.size());
    }
};

REG_FACTORY_FOR(ImplFactory<ArgMaxImpl>, ArgMax);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
