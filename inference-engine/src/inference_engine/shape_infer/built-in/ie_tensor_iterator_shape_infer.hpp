// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <description_buffer.hpp>
#include "ie_built_in_impl.hpp"
#include <shape_infer/ie_reshaper.hpp>
#include <ie_layers.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Shape inference for DetectionOutput layer
 */
class TensorIteratorShapeProp : public BuiltInShapeInferImpl {
public:
    explicit TensorIteratorShapeProp(const std::string& type) : BuiltInShapeInferImpl(type) {}

    void setOriginalLayer(const CNNLayer *layer) {
        auto ti = dynamic_cast<const TensorIterator*>(layer);
        if (!ti)
            THROW_IE_EXCEPTION << "Error during shape infer. Original layer is not TensorIterator.";
        _original_ti = ti;
    }

    void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                         const std::map<std::string, std::string>& params,
                         const std::map<std::string, Blob::Ptr>& blobs,
                         std::vector<SizeVector>& outShapes) override {
        LayerParams lp{};
        TensorIterator ti(lp);
        ti.params = params;
        ti.type = _type;
        ti.body = _original_ti->body;
        ti.back_edges = _original_ti->back_edges;
        ti.input_port_map = _original_ti->input_port_map;
        ti.output_port_map = _original_ti->output_port_map;
        validate(&ti, inBlobs, params, blobs);

        // TODO: make util function to calculate num of iteration
        int num_iteration = 1;

        // Prepare input shapes for internal body
        std::map<std::string, std::vector<size_t>> newInShapes;
        for (auto &port_map : ti.input_port_map) {
            int ext_port = port_map.from;
            int int_port = port_map.to;
            auto int_name = ti.body.inputs[int_port]->name;

            auto shape = inShapes[ext_port];
            if (port_map.axis != -1) {
                int size = shape[port_map.axis];
                int start = port_map.start < 0
                        ? port_map.start + size + 1
                        : port_map.start;
                int end = port_map.end < 0
                        ? port_map.end + size + 1
                        : port_map.end;

                num_iteration = std::abs(end - start) / std::abs(port_map.stride);

                // port with iterating through. Change dimension with iteration
                shape[port_map.axis] = port_map.part_size;
            }

            newInShapes[int_name] = shape;
        }

        // Body shape infer
        _body_reshaper = std::make_shared<Reshaper>(_original_ti->body.inputs);
        _body_reshaper->runNoApply(newInShapes);

        outShapes.resize(ti.output_port_map.size());
        for (auto &port_map : ti.output_port_map) {
            int ext_port = port_map.from;
            int int_port = port_map.to;
            auto &int_out_data = ti.body.outputs[int_port];
            auto shape = _body_reshaper->getResultShapeFor(int_out_data);

            if (port_map.axis != -1) {
                // port with iterating through. Change dimension with iteration
                shape[port_map.axis] *= num_iteration;
            }

            outShapes[ext_port] = shape;
        }
    }

    void apply() {
        if (!_body_reshaper)
            THROW_IE_EXCEPTION << "Request of apply reshape results while shape infer was not finished";
        _body_reshaper->apply();
    }


private:
    const TensorIterator* _original_ti = nullptr;
    std::shared_ptr<Reshaper> _body_reshaper;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
