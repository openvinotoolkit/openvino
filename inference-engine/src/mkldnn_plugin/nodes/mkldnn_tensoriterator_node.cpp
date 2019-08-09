// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tensoriterator_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <ie_layers_internal.hpp>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_memcpy.h>
#include "details/caseless.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

static LayerConfig make_plain_config(const CNNLayerPtr &layer) {
    LayerConfig config;

    for (const auto &in_w : layer->insData) {
        const auto in = in_w.lock();

        const auto dims = in->getDims();
        const auto prec = in->getPrecision();

        DataConfig data_conf {};
        data_conf.desc = TensorDesc { prec, dims, TensorDesc::getLayoutByDims(dims) };
        config.inConfs.push_back(data_conf);
    }

    for (const auto &out : layer->outData) {
        const auto dims = out->getDims();
        const auto prec = out->getPrecision();

        DataConfig data_conf {};
        data_conf.desc = TensorDesc { prec, dims, TensorDesc::getLayoutByDims(dims) };
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to,
            bool as_input, const TensorIterator::PortMap &port_map, const mkldnn::engine& eng, int n_iter) : as_input(as_input) {
        const auto &full_blob = as_input ? from : to;
        const auto &part_blob = !as_input ? from : to;

        auto axis = port_map.axis;
        auto stride = port_map.stride;

        auto full_dims = full_blob->GetDims();
        auto part_dims = part_blob->GetDims();

        bool simple_copy = port_map.axis == -1;
        if (port_map.axis == -1) {
            // simple copy mode. No iteration through this tensor
            reorders.emplace_back(from->GetPrimitive(), to->GetPrimitive());
            iter_count = n_iter;
        } else {
            auto abs_stride = std::abs(stride);
            auto sign_of_stride = stride < 0.0f ? -1 : 1;

            IE_ASSERT(n_iter == full_dims[axis] / abs_stride) << "Shape mismatch for tensor iterator port";

            full_dims[axis] = abs_stride;
            IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

            iter_count = n_iter;

            // make chunk view
            auto chunk_desc =  full_blob->GetDescriptor();
            chunk_desc.data.dims[axis] = 1;
            chunk_desc.data.layout_desc.blocking.padding_dims[axis] = 1;  // TODO: asamption that plain tensor

            mem_holder.push_back(full_blob->GetPrimitive());
            auto full_mem_handler = full_blob->GetPrimitive().get_data_handle();
            mem_holder.emplace_back(mkldnn::memory::primitive_desc(chunk_desc, eng), full_mem_handler);
            auto &chunk_mem_prim = mem_holder.back();

            auto elem_size = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(chunk_desc.data.data_type));

            // TODO: only stride 1
            chunk_stride_in_byte = chunk_desc.data.layout_desc.blocking.strides[0][axis] * elem_size;
            chunk_offset_in_byte = sign_of_stride < 0 ? (iter_count - 1) * chunk_stride_in_byte : 0;
            chunk_stride_in_byte *= sign_of_stride;

            if (as_input) {
                reorders.emplace_back(chunk_mem_prim, to->GetPrimitive());
            } else {
                reorders.emplace_back(from->GetPrimitive(), chunk_mem_prim);
            }
        }
    }

    void execute(int n_iter, mkldnn::stream strm) override {
        if (chunk_stride_in_byte != 0) {
            IE_ASSERT(n_iter < iter_count);

            auto full_mem = mem_holder[FULL_DATA];
            auto chunk_mem = mem_holder[CHUNK_DATA];

            chunk_mem.set_data_handle(static_cast<uint8_t *>(full_mem.get_data_handle()) +
                    chunk_offset_in_byte + chunk_stride_in_byte * n_iter);

            strm.submit({reorders.begin(), reorders.end()});
        } else {
            if (as_input ? n_iter == 0 : n_iter == (iter_count - 1))
                strm.submit({reorders.begin(), reorders.end()});
        }
    };

private:
    bool as_input;
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    const int FULL_DATA = 0;
    const int CHUNK_DATA = 1;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng, int n_iter) {
        auto mem_desc =  from->GetDescriptor();


        mem_holder.emplace_back(mkldnn::memory::primitive_desc(mem_desc, eng));
        auto &temp_mem = mem_holder.back();

        reorders.emplace_back(from->GetPrimitive(), to->GetPrimitive());

        iter_count = n_iter;
    }

    void execute(int n_iter, mkldnn::stream strm) override {
        if (n_iter < iter_count - 1) {
            strm.submit({reorders.begin(), reorders.end()});
        }
    };
};

}  // namespace MKLDNNPlugin

MKLDNNTensorIteratorNode::MKLDNNTensorIteratorNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket) :
        MKLDNNNode(layer, eng, socket) {}

void MKLDNNTensorIteratorNode::getSupportedDescriptors() {
    auto *ti = dynamic_cast<class TensorIterator*>(getCnnLayer().get());
    if (ti == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert to TensorIterator layer.";

    n_iter = getNumIteration(*ti);
    MKLDNNGraph::ApplyUnrollPasses(ti->body);
    sub_graph.CreateGraph(ti->body, ext_mng, this->whichSocket());

    // Try to detect inputs and outputs by indexes
    std::map<std::string, MKLDNNNodePtr> in_map, out_map;
    for (auto node : sub_graph.GetNodes())
        if (node->getType() == Input)  // filter by type Input
            in_map[node->getName().substr(3)] = node;  // remove "in_" prefix

    for (auto node : sub_graph.GetOutputNodes())
        out_map[node->getName().substr(4)] = node;  // remove "out_" prefix

    for (const auto &in_data : ti->body.inputs) {
        if (in_data->getName() == "const_holder") continue;

        auto &in_node = in_map[in_data->getName()];
        auto in_mem = in_node->getChildEdgeAt(0)->getMemoryPtr();
        input_mem.push_back(in_mem);
    }

    for (const auto &out_data : ti->body.outputs) {
        auto &out_node = out_map[out_data->getName()];
        auto out_mem = out_node->getParentEdgeAt(0)->getMemoryPtr();
        output_mem.push_back(out_mem);
    }
}

void MKLDNNTensorIteratorNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto config = make_plain_config(getCnnLayer());
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}


void MKLDNNTensorIteratorNode::createPrimitive() {
    auto ti = dynamic_cast<class TensorIterator*>(getCnnLayer().get());
    if (ti == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert to TensorIterator layer.";

    for (auto map_rule : ti->input_port_map) {
        auto &extr_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &intr_mem = input_mem[map_rule.to];

        auto mapper = std::shared_ptr<PortMapHelper>(
                new PortIteratorHelper (extr_mem, intr_mem, true, map_rule, getEngine(), n_iter));

        in_port_mappers.push_back(mapper);
    }

    for (auto map_rule : ti->output_port_map) {
        auto &extr_mem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &intr_mem = output_mem[map_rule.to];

        auto mapper = std::shared_ptr<PortMapHelper>(
                new PortIteratorHelper (intr_mem, extr_mem, false, map_rule, getEngine(), n_iter));

        out_port_mappers.push_back(mapper);
    }

    for (auto map_rule : ti->back_edges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mem = input_mem[map_rule.to];

        auto mapper = std::shared_ptr<PortMapHelper>(
                new BackEdgePortHelper(from_mem, to_mem, getEngine(), n_iter));

        out_port_mappers.push_back(mapper);
    }
}

void MKLDNNTensorIteratorNode::execute(mkldnn::stream strm) {
    sub_graph.ResetInferCount();

    for (int i = 0; i < n_iter; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : in_port_mappers)
            mapper->execute(i, strm);

        sub_graph.Infer();

        // copy data from subgraph iteration to outputs
        // or next iteration inputs
        for (auto &mapper : out_port_mappers)
            mapper->execute(i, strm);
    }
}

bool MKLDNNTensorIteratorNode::created() const {
    return getType() == TensorIterator;
}
