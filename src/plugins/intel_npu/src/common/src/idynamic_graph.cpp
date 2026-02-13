// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include "intel_npu/common/idynamic_graph.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

void IDynamicGraph::MemRefType::setArg(const void* arg) {
    _basePtr = _data = arg;
}

void IDynamicGraph::MemRefType::setSize(const ov::Shape& shape) {
    // Note: check difference between shape from compiler and shape from IR.
    _sizes.resize(shape.size());
    _strides.resize(shape.size());
    _dimsCount = static_cast<uint32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        _sizes[i] = shape[i];
        _strides[i] = 0;  // Strides are not set yet, will be updated
    }
}

void IDynamicGraph::MemRefType::set(const void* arg, int64_t offset, std::shared_ptr<ov::ITensor> tensor) {
    _basePtr = _data = arg;
    _offset = offset;
    auto& shape = tensor->get_shape();
    for (size_t j = 0; j < shape.size(); j++) {
        _sizes[j] = shape[j];
    }
    auto& strides = tensor->get_strides();
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    for (size_t j = 0; j < strides.size(); j++) {
        _strides[j] = strides[j] / elementSize;
    }
    _dimsCount = shape.size();
}

void IDynamicGraph::MemRefType::updateStride() {
    // Note: NCHW layout style
    uint64_t stride = 1;
    for (int64_t i = _dimsCount - 1; i >= 0; --i) {
        _strides[i] = stride;
        stride *= _sizes[i];
    }
}

// The comparision only checks shape and strides now
bool IDynamicGraph::MemRefType::compare(const IDynamicGraph::MemRefType& memref) {
    if (memref._dimsCount != _dimsCount || _sizes.size() != memref._sizes.size() ||
        _strides.size() != memref._strides.size())
        return false;
    size_t dimsCount = static_cast<size_t>(_dimsCount);
    if (memref._sizes.size() != dimsCount || memref._strides.size() != dimsCount)
        return false;
    for (size_t i = 0; i < dimsCount; i++) {
        if (_sizes[i] != memref._sizes[i] || _strides[i] != memref._strides[i]) {
            return false;
        }
    }
    return true;
}

std::ostream& operator<<(std::ostream& os, const IDynamicGraph::MemRefType& memRef) {
    os << "BasePtr: " << memRef._basePtr << ", Data: " << memRef._data << ", Offset: " << memRef._offset
       << ", Sizes: [";
    for (int64_t size : memRef._sizes) {
        os << size << " ";
    }
    os << "], Strides: [";
    for (int64_t stride : memRef._strides) {
        os << stride << " ";
    }
    os << "]";
    return os;
}

std::string IDynamicGraph::MemRefType::toString() {
    std::stringstream stream;
    stream << *this;
    return stream.str();
}

void IDynamicGraph::GraphArguments::setArgumentValue(uint32_t argi, const void* argv) {
    if (argi < _inputs.size()) {
        _inputs[argi]._basePtr = _inputs[argi]._data = const_cast<void*>(argv);
    } else {
        auto idx = argi - _inputs.size();
        if (idx < _outputs.size()) {
            _outputs[idx]._basePtr = _outputs[idx]._data = const_cast<void*>(argv);
        }
    }
}

void IDynamicGraph::GraphArguments::setArgumentProperties(uint32_t argi,
                                                          const void* argv,
                                                          const ov::Shape& sizes,
                                                          const std::vector<size_t>& strides) {
    if (argi < _inputs.size()) {
        _inputs[argi]._basePtr = _inputs[argi]._data = const_cast<void*>(argv);
        for (int64_t i = 0; i < _inputs[argi]._dimsCount; i++) {
            _inputs[argi]._sizes[i] = sizes[i];
            _inputs[argi]._strides[i] = strides[i];
        }
    } else {
        auto idx = argi - _inputs.size();
        if (idx < _outputs.size()) {
            _outputs[idx]._basePtr = _outputs[idx]._data = const_cast<void*>(argv);
            for (int64_t i = 0; i < _outputs[idx]._dimsCount; i++) {
                _outputs[idx]._sizes[i] = sizes[i];
                _outputs[idx]._strides[i] = strides[i];
            }
        }
    }
}

void IDynamicGraph::execute(const std::shared_ptr<ZeroInitStructsHolder>&,
                            GraphArguments&,
                            std::vector<ze_command_list_handle_t>&,
                            ze_command_queue_handle_t,
                            ze_fence_handle_t,
                            ze_event_handle_t,
                            ze_graph_profiling_pool_handle_t) {
    OPENVINO_THROW("execute not implemented");
}

void IDynamicGraph::getBinding(GraphArguments&) {
    OPENVINO_THROW("getBinding not implemented");
}

uint64_t IDynamicGraph::get_num_subgraphs() const {
    OPENVINO_THROW("get_num_subgraphs not implemented");
}

void IDynamicGraph::predict_output_shape(std::vector<MemRefType>&, std::vector<MemRefType>&) {
    OPENVINO_THROW("predict_output_shape not implemented");
}

}  // namespace intel_npu
