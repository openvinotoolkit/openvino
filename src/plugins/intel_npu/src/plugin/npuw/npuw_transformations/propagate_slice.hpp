// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// rt_info key set by PropagateSliceThroughSDPA (rule R3) on the cloned SDPA node when it
// slices Q's sequence axis: it records the query length *before* slicing, since K/V (and
// thus context length) still reflect the original length. Its mere presence signals that
// the slice propagation happened. Consumed by HFA and pyramid-attention extraction.
inline constexpr const char* NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY = "npuw_original_query_length";

// Read the original (pre-slice) query length stashed by PropagateSliceThroughSDPA on
// the SDPA's second MatMul node. Returns std::nullopt if the node is null, the rt_info
// is absent, or it cannot be read as size_t.
std::optional<std::size_t> find_propagated_original_query_length(const std::shared_ptr<ov::Node>& matmul2_node);

// Resolve the original (pre-slice) query length for an SDPA subgraph, shared by both
// HFA and pyramid-attention extraction. Prefers the value stashed by PropagateSliceUp on
// `matmul2_node` (see find_propagated_original_query_length above); falls back to
// `fallback_length` (typically derived from Q's or Softmax's static shape) when no such
// rt_info is present, i.e. PropagateSliceUp never touched this SDPA.
std::size_t resolve_original_query_length(std::size_t fallback_length, const std::shared_ptr<ov::Node>& matmul2_node);

// Propagates an ov::op::v8::Slice upstream through elementwise ops and SDPA.
// Each rule matches Slice(SomeOp(...)) and replaces it with SomeOp(Slice(...)).
// GraphRewrite fires all rules to convergence so the Slice moves as far upstream
// as possible in one pass invocation.
//
// Preconditions checked by every rule:
//   1. The parent op has exactly one output consumer (the Slice).
//   2. The Slice actually reduces the tensor (input_size > output_size on the sliced axis).
//
// Rules implemented:
//   R1. Slice(Eltwise1(X))          -> Eltwise1(Slice(X))
//       (single-input elementwise: Gelu, Relu, Sqrt, Tanh, Sigmoid, Erf, Convert, ...)
//       For Convert, the destination element type is explicitly preserved.
//   R2a. Slice(Eltwise2(A, B))      -> Eltwise2(Slice(A), Slice(B))
//        when A.shape == B.shape
//   R2b. Slice(Eltwise2(A, B))      -> Eltwise2(Slice(A), B)   [or symmetric]
//        when one operand broadcasts on the sliced axis
//   R3. Slice(SDPA(Q,K,V,mask,...)) -> SDPA(Slice(Q),K,V,Slice(mask),...)
//        when the sliced axis is the Q sequence dimension
//   R4. Slice(Reduce*(X, axis))     -> Reduce*(Slice(X), axis)
//        when the sliced axis is different from the reduction axis
//        (supports ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd, etc.)
//   R5. Slice(MatMul(X, W))         -> MatMul(Slice(X), W)
//        when the sliced axis is not the feature (last) dimension
//   R6. Slice(Reshape(X))           -> Reshape(Slice(X))
//        when the sliced axis structure is preserved by Reshape
//   R7. Slice(Transpose(X))         -> Transpose(Slice(X))
//        mapping slice axis through the permutation
//   R8. Slice(VariadicSplit(X)[i]) for all i -> VariadicSplit(Slice(X))
//        when all outputs have identical Slice consumers on the non-split axis
//   R9. Merge duplicate Slice nodes with identical inputs and parameters
//   R10. Slice(Reshape(Tile(X)))    -> Reshape(Tile(Slice(X)))
//        when Tile expands a dimension that Reshape then splits, and Slice operates on the split dimension
//        (e.g., Tile([1024,2048],[128,1]) -> Reshape([131072,2048]) -> [128,1024,2048] -> Slice(axis=1))
//   R11. Slice(Unsqueeze(X))        -> Unsqueeze(Slice(X))
//        mapping slice axis accounting for inserted dimensions
//        (e.g., Unsqueeze([128,1024],axes=[2]) -> [128,1024,1] -> Slice(axis=1) -> [128,1,1])
//   R12. Slice(ScatterElementsUpdate(data, indices, updates)) ->
//        ScatterElementsUpdate(Slice(data), Slice(indices), Slice(updates))
//        when the sliced axis is not the scatter axis
//        (e.g., ScatterElementsUpdate([1024,8],[1024,8],[1024,1],axis=1) -> [1024,128] -> Slice(axis=0) -> [1,128])
//   R13. Slice(Broadcast(X)) -> Broadcast(X) with adjusted target shape
//        when input shape is compatible with slice output
//        (e.g., Broadcast([1],[1024,128]) -> [1024,128] -> Slice(axis=0) -> [1,128])
//   R14. Remove no-op Slice nodes where input_shape == output_shape
//   R15. Slice(TopK(X)[0]), Slice(TopK(X)[1]) -> TopK(Slice(X))
//        when both TopK outputs (values and indices) are consumed by semantically equivalent Slices
//        and the slice axis != TopK axis (otherwise would change TopK result)
//        (e.g., TopK([1024,128],axis=1,k=8) -> [1024,8] -> Slice(axis=0) -> [1,8])
//   R16. Slice(Softmax(X, axis=A), axis=B) -> Softmax(Slice(X, axis=B), axis=A)
//        when slice axis != Softmax axis (otherwise would change Softmax result)
//        (e.g., Softmax([1024,128],axis=1) -> [1024,128] -> Slice(axis=0) -> [1,128])
//   R17. Slice(Concat(X1,X2,...,axis=A), axis=B) -> Concat(Slice(X1,axis=B),Slice(X2,axis=B),...,axis=A)
//        when slice axis != Concat axis (otherwise would change Concat result)
//        (e.g., Concat([1,32,1024,64],[1,32,1024,64],axis=3) -> [1,32,1024,128] -> Slice(axis=2) -> [1,32,1,128])
//   R17a. Slice(Gather(X, indices, axis=A), axis=B) -> Gather(Slice(X, axis=B), indices, axis=A)
//         when slice axis != Gather axis (otherwise would affect the gathered dimension)
//         (e.g., Gather([1,1024,35,256],axis=2) -> [1,1024,256] -> Slice(axis=1) -> [1,1,256]
//          becomes Slice([1,1024,35,256],axis=1) -> [1,1,35,256] -> Gather(axis=2) -> [1,1,256])
//   R18. Slice(Slice(X)) -> Single Slice with merged parameters
//        when the two slices operate on different axes (same axis composition is complex, skipped for now)
//        (e.g., Slice([1,32,1024,128],axis=3) -> [1,32,1024,64] -> Slice(axis=2) -> [1,32,1,64]
//         becomes Slice([1,32,1024,128],axes=[2,3]) -> [1,32,1,64])
//   R19. Extract common slice axes before Transpose when multiple Slices consume it
//        Pattern: Transpose -> [Slice1, Slice2, ...] with common slice axes
//        Result: Slice(common axes) -> Transpose -> [residual_Slice1, residual_Slice2, ...]
//        (e.g., Transpose([1,1024,32,128]->[1,32,1024,128]) -> Slice1(axes=[2,3]), Slice2(axes=[2,3]),
//        Slice3(axes=[2,3])
//         where all have axis=2: 1024->1 (same), but axis=3 differs
//         becomes Slice(axis=1: 1024->1) -> Transpose([1,1,32,128]->[1,32,1,128]) -> residual slices on axis=3 only)
//   R20. Extract common slice axes before Binary op (Add/Multiply/etc.) when multiple Slices consume it
//        Pattern: Binary(A,B) -> [Slice1, Slice2, ...] with common slice axes
//        Result: Binary(Slice(A),Slice(B)) -> [residual_Slice1, residual_Slice2, ...]
//        (e.g., Multiply([1,1,1,512],[1,1024,8,512])->[1,1024,8,512] -> Slice1(axes=[1,3]), Slice2(axes=[1,3]),
//        Slice3(axes=[1])
//         where all have axis=1: 1024->1 (same), but axis=3 differs
//         becomes Multiply([1,1,1,512],Slice([1,1024,8,512]->[1,1,8,512]))->[1,1,8,512] -> residual slices on axis=3
//        only)

class PropagateSliceUp : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PropagateSliceUp");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
