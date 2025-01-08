// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference.hpp"

namespace ov {
namespace snippets {

bool broadcast_merge_into(VectorDims& dst, const VectorDims& src, const ov::op::AutoBroadcastSpec& autob = ov::op::AutoBroadcastType::NUMPY);

bool merge_into(VectorDims& dst, const VectorDims& src);

class NumpyBroadcastShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};


template<class BroadcastOP>
class BroadcastShapeInfer : public IShapeInferSnippets {
    std::shared_ptr<BroadcastOP> broadcast_op;
public:
    explicit BroadcastShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class PassThroughShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(!input_shapes.empty(), "Empty Input shapes are not allowed for PassThroughShapeInfer");
        return {{input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

class EmptyShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{}, ShapeInferStatus::success};
    }
};

class SingleElementShapeInfer : public IShapeInferSnippets {
public:
    inline Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        return {{{1}}, ShapeInferStatus::success};
    }
};

class SelectShapeInfer : public IShapeInferSnippets {
    ov::op::AutoBroadcastSpec m_broadcast_spec;
public:
    explicit SelectShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class HorizonOpShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class BrgemmShapeInfer : public IShapeInferSnippets {
    std::vector<std::vector<size_t>> m_io_layouts;
public:
    explicit BrgemmShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class ReduceShapeInfer : public IShapeInferSnippets {
    size_t m_axis;
public:
    explicit ReduceShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

} // namespace snippets
} // namespace ov
