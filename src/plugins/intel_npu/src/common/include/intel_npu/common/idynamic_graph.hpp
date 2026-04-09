// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

class IDynamicGraph : public IGraph {
public:
    struct MemRefType {
        const void* _basePtr;
        const void* _data;
        int64_t _offset;
        std::vector<int64_t> _sizes;
        std::vector<int64_t> _strides;
        int64_t _dimsCount;
        std::shared_ptr<void> _impl;

        MemRefType() : _basePtr(nullptr), _data(nullptr), _offset(0), _sizes(), _strides(), _dimsCount(0) {}

        MemRefType(const void* basePtr,
                   const void* data,
                   int64_t offset,
                   const std::vector<int64_t>& sizes,
                   const std::vector<int64_t>& strides,
                   int64_t dimsCount)
            : _basePtr(basePtr),
              _data(data),
              _offset(offset),
              _sizes(sizes),
              _strides(strides),
              _dimsCount(dimsCount) {}

        void setArg(const void* arg);
        void setSize(const ov::Shape& shape);
        void set(const void* basePtr, int64_t offset, std::shared_ptr<ov::ITensor> tensor);
        void updateStride();
        bool compare(const MemRefType& memref);
        friend std::ostream& operator<<(std::ostream& os, const IDynamicGraph::MemRefType& memRef);
        std::string toString();
    };

    struct GraphArguments {
        std::vector<MemRefType> _inputs;
        std::vector<MemRefType> _outputs;
        std::shared_ptr<void> _impl;

        void setArgumentValue(uint32_t argi, const void* argv);
        void setArgumentProperties(uint32_t argi,
                                   const void* argv,
                                   const ov::Shape& shapes,
                                   const std::vector<size_t>& strides);
    };

    IDynamicGraph() = default;
    ~IDynamicGraph() override = default;

    /**
     * @brief Returns the opaque VM engine handle owned by the graph.
     *
     * The pipeline casts this to @c npu_vm_runtime_handle_t and passes it to
     * @c vm_execute_graph() so that the graph object does not need to hold
     * inference-request-specific state.
     */
    virtual void* get_vm_engine() const {
        return nullptr;
    }

    virtual void getBinding(GraphArguments& args);

    virtual uint64_t get_num_subgraphs() const;

    virtual void predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                                      std::vector<MemRefType>& outputDescriptors);
};

}  // namespace intel_npu
