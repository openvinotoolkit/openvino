// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov::snippets {

bool broadcast_merge_into(VectorDims& dst,
                          const VectorDims& src,
                          const ov::op::AutoBroadcastSpec& autob = ov::op::AutoBroadcastType::NUMPY);

bool merge_into(VectorDims& dst, const VectorDims& src);

class NumpyBroadcastShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

template <class BroadcastOP>
class BroadcastShapeInfer : public IShapeInferSnippets {
    std::shared_ptr<BroadcastOP> broadcast_op;

public:
    explicit BroadcastShapeInfer(const std::shared_ptr<Node>& n);
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
};

class PassThroughShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(!input_shapes.empty(), "Empty Input shapes are not allowed for PassThroughShapeInfer");
        return {{input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

class EmptyShapeInfer : public IShapeInferSnippets {
public:
    Result infer([[maybe_unused]] const std::vector<VectorDimsRef>& input_shapes) override {
        return {{}, ShapeInferStatus::success};
    }
};

class SingleElementShapeInfer : public IShapeInferSnippets {
public:
    Result infer([[maybe_unused]] const std::vector<VectorDimsRef>& input_shapes) override {
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

class OnlineSoftmaxUpdateMaxShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of shapes to OnlineSoftmaxUpdateMaxShapeInfer.");
        return {{input_shapes[0].get(), input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

class OnlineSoftmaxUpdateSumShapeInfer : public IShapeInferSnippets {
public:
    Result infer(const std::vector<VectorDimsRef>& input_shapes) override {
        OPENVINO_ASSERT(input_shapes.size() == 2, "Invalid number of shapes to OnlineSoftmaxUpdateSumShapeInfer.");
        return {{input_shapes[0].get(), input_shapes[0].get()}, ShapeInferStatus::success};
    }
};

}  // namespace ov::snippets
