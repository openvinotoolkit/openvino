// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tensoriterator_node.h"
#include "desc_iterator.hpp"
#include <legacy/ie_layers.h>
#include <legacy/ie_layers_internal.hpp>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

static InferenceEngine::LayerConfig make_plain_config(const InferenceEngine::CNNLayerPtr &layer) {
    using namespace InferenceEngine;

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
    PortIteratorHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, bool sliced_src,
                       const InferenceEngine::TensorIterator::PortMap &slice_rule, const mkldnn::engine& eng) {
        const auto &full_blob = sliced_src ? from : to;
        const auto &part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.axis;
        auto stride = slice_rule.stride;

        auto full_dims = full_blob->GetDims();
        auto part_dims = part_blob->GetDims();

        auto abs_stride = std::abs(stride);
        auto sign_of_stride = stride < 0.0f ? -1 : 1;

        iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

        // make chunk view
        auto chunk_desc =  full_blob->GetDescriptor();
        chunk_desc.data.dims[axis] = abs_stride;
        chunk_desc.data.layout_desc.blocking.padding_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        mem_holder.push_back(full_blob->GetPrimitive());
        auto full_mem_handler = full_blob->GetPrimitive().get_data_handle();
        mem_holder.emplace_back(mkldnn::memory::primitive_desc(chunk_desc, eng), full_mem_handler);
        auto &chunk_mem_prim = mem_holder.back();

        auto elem_size = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(chunk_desc.data.data_type));

        chunk_stride_in_byte = chunk_desc.data.layout_desc.blocking.strides[0][axis] * elem_size * abs_stride;
        chunk_offset_in_byte = sign_of_stride < 0 ? (iter_count - 1) * chunk_stride_in_byte : 0;
        chunk_stride_in_byte *= sign_of_stride;

        if (sliced_src) {
            reorders.emplace_back(chunk_mem_prim, to->GetPrimitive());
        } else {
            reorders.emplace_back(from->GetPrimitive(), chunk_mem_prim);
        }
    }

    void execute(mkldnn::stream strm, int iter) override {
        IE_ASSERT(iter >= 0 && iter < iter_count);

        auto full_mem = mem_holder[FULL_DATA];
        auto chunk_mem = mem_holder[CHUNK_DATA];

        chunk_mem.set_data_handle(static_cast<uint8_t *>(full_mem.get_data_handle()) +
                chunk_offset_in_byte + chunk_stride_in_byte * iter);

        strm.submit({reorders.begin(), reorders.end()});
    }

private:
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    const int FULL_DATA = 0;
    const int CHUNK_DATA = 1;
    int iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        reorders.emplace_back(from->GetPrimitive(), to->GetPrimitive());
    }

    void execute(mkldnn::stream strm, int iter) override {
        if (iter != 0) {
            strm.submit({reorders.begin(), reorders.end()});
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
        // Only scalar I32 tensor is supported
        IE_ASSERT(to->GetDataType() == memory::s32);
        IE_ASSERT(to->GetDims() == memory::dims{1});
        mem_holder.push_back(to->GetPrimitive());
    }

    void execute(mkldnn::stream strm, int n_iter) override {
        auto mem = mem_holder[0];
        auto data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        *data_ptr = n_iter;
    }
};

class asBoolCheck : public PortChecker {
public:
    asBoolCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::u8);
        IE_ASSERT(mem->GetDims() == memory::dims{1});
        mem_holder.push_back(mem->GetPrimitive());
    }

    int getStatus() override {
        auto mem = mem_holder[0];
        auto data_ptr = static_cast<uint8_t*>(mem.get_data_handle());
        return *data_ptr == static_cast<uint8_t>(0) ? 0 : 1;
    }
};

class asIntCheck : public PortChecker {
public:
    asIntCheck(const MKLDNNMemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::s32);
        IE_ASSERT(mem->GetDims() == memory::dims{1});
        mem_holder.push_back(mem->GetPrimitive());
    }

    int getStatus() override {
        auto mem = mem_holder[0];
        auto data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        return *data_ptr;
    }
};

class staticValueCheck : public PortChecker {
public:
    staticValueCheck(const int &value) : value(value) {}

    int getStatus() override {
        return value;
    }
private:
    int value;
};

}  // namespace MKLDNNPlugin

MKLDNNTensorIteratorNode::MKLDNNTensorIteratorNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNTensorIteratorNode::getSupportedDescriptors() {
    auto *ti = dynamic_cast<class InferenceEngine::TensorIterator*>(getCnnLayer().get());
    if (ti == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert to TensorIterator layer.";

    n_iter = getNumIteration(*ti);
    sub_graph.CreateGraph(ti->body, ext_mng, weightCache);

    // Try to detect inputs and outputs by indexes
    const auto &in_map = sub_graph.GetInputNodes();
    for (const auto &in_data : ti->body.inputs) {
        if (in_data->getName() == "const_holder") continue;

        auto &in_node = in_map.at(in_data->getName());
        auto in_mem = in_node->getChildEdgeAt(0)->getMemoryPtr();
        input_mem.push_back(in_mem);
    }

    // Assume that order of outputs in original TI and produces sub_graph is same
    const auto &out_vec = sub_graph.GetOutputNodes();
    for (size_t i = 0; i < out_vec.size(); i++) {
        auto out_mem = out_vec[i]->getParentEdgeAt(0)->getMemoryPtr();
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
    auto ti = dynamic_cast<class InferenceEngine::TensorIterator*>(getCnnLayer().get());
    if (ti == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert to TensorIterator layer.";

    const auto &eng = getEngine();

    for (auto map_rule : ti->input_port_map) {
        auto &from_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &to_mem = input_mem[map_rule.to];

        if (map_rule.axis == -1)
            first_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
        else
            before_mappers.emplace_back(new PortIteratorHelper(from_mem, to_mem, true, map_rule, eng));
    }

    for (auto map_rule : ti->output_port_map) {
        auto &to_mem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &from_mem = output_mem[map_rule.to];

        if (map_rule.axis == -1)
            last_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
        else
            after_mappers.emplace_back(new PortIteratorHelper(from_mem, to_mem, false, map_rule, eng));
    }

    for (auto map_rule : ti->back_edges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mem = input_mem[map_rule.to];

        before_mappers.emplace_back(new BackEdgePortHelper(from_mem, to_mem, eng));
    }

    // special purpose ports
    constexpr auto key_cur_iter_port = "loop_body_current_iteration_idx";
    constexpr auto key_cond_port = "loop_body_condition_output_idx";
    constexpr auto key_trip_count_port = "loop_trip_count_idx";
    constexpr auto key_init_cond_port = "loop_execution_condition_idx";

    auto iter_idx_ports = ti->GetParamAsInts(key_cur_iter_port, {});
    for (auto idx : iter_idx_ports) {
        auto to_mem = input_mem[idx];
        before_mappers.emplace_back(new IterCountPortHelper(to_mem, eng));
    }

    auto condition_port_idx = ti->GetParamAsInt(key_cond_port, -1);
    if (condition_port_idx == -1) {
        continue_cond_check.reset(new staticValueCheck(true)); // always true
    } else {
        auto mem = output_mem[condition_port_idx];
        continue_cond_check.reset(new asBoolCheck(mem));
    }

    auto trip_count_port_idx = ti->GetParamAsInt(key_trip_count_port, -1);
    if (trip_count_port_idx == -1) {
        trip_count_check.reset(new staticValueCheck(n_iter)); // use statically calculated num of iteration
    } else {
        auto mem = getParentEdgesAtPort(trip_count_port_idx)[0]->getMemoryPtr();
        trip_count_check.reset(new asIntCheck(mem));
    }

    auto init_cond_port_idx = ti->GetParamAsInt(key_init_cond_port, -1);
    if (init_cond_port_idx == -1) {
        initial_cond_check.reset(new staticValueCheck(true));
    } else {
        auto mem = getParentEdgesAtPort(init_cond_port_idx)[0]->getMemoryPtr();
        initial_cond_check.reset(new asBoolCheck(mem));
    }
}

void MKLDNNTensorIteratorNode::execute(mkldnn::stream strm) {
    sub_graph.ResetInferCount();

    bool continue_cond = initial_cond_check->getStatus();
    int max_num_iter = trip_count_check->getStatus();

    for (auto &mapper : first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : before_mappers)
            mapper->execute(strm, i);

        sub_graph.Infer();

        continue_cond = continue_cond_check->getStatus();

        // copy data from subgraph iteration to outputs
        // or to next iteration inputs
        for (auto &mapper : after_mappers)
            mapper->execute(strm, i);
    }

    for (auto &mapper : last_mappers)
        mapper->execute(strm);
}

bool MKLDNNTensorIteratorNode::created() const {
    return getType() == TensorIterator;
}
REG_MKLDNN_PRIM_FOR(MKLDNNTensorIteratorNode, TensorIterator);
