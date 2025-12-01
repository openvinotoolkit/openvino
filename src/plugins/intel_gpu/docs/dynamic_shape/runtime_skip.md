# Runtime operation skip
## Description
When working with dynamic shapes, compilation-time optimization faces inherent limitations since shape information remains undefined until runtime. This creates a two-phase optimization opportunity: while certain operations cannot be optimized during the initial compilation phase due to unknown shapes, they become prime candidates for runtime optimization once concrete shape information materializes during inference execution. 

Consider a 4D permute operation with the transformation order [0, 2, 1, 3]. During compilation, the input shapes are dynamic [-1, -1, -1, -1], therefore, any shape-based optimization is not applicable. However, there might be a second chance to optimize this operation during the runtime. Suppose the actual input shape resolves to [128, 1, 32, 64]. With this concrete information, the we can now recognize a critical insight: since dimension 1 has size 1, swapping dimensions 1 and 2 (as specified by the permute order [0, 2, 1, 3]) results in no actual data movement. The operation becomes essentially a metadata-only transformation—a simple reshape that requires no memory copying or data rearrangement.
This example demonstrates how runtime optimization can transform potentially expensive operations to be skipped, highlighting the value of deferred optimization strategies in dynamic computation graphs.

## Basic flow of runtime operation skip
1. **Relevant flags**
First, we need to set two flags for the program_node of such an operation, which we do not apply shape-based optimization during compilation but try runtime optimization with the shape.
- Static flags (Set during `mark_runtime_skippable_nodes` pass at compilation time)
  - `program_node::optimized`
    - This flag presents that this node is eligible for being optimized out, either at compilation time or runtime.
    - This flag is set true for all optimization schemes, not limited to runtime skippability.
  - `program_node::runtime_skippable`
    - Indicates that this node can be optimized during runtime based on the shape.
- Dynamic flag (Set at runtime)
  - `primitive_inst::_can_be_optimized`
     - Indicates that this `primitive_inst` is actually optimized out at a certain execution
  
If `program_node::optimized` is true and `program_node::runtime_skippable` is false, it means that this node is *always* optimized out (i.e., compile-time optimization).
If both of the flags are set true, the node may be optimized out or not in the runtime, depending on the runtime shapes.
If program_node::optimized is false and program_node::runtime_skippable is true, it is an invalid combination.

As an example of using both flags,  please refer to [memory_dependency_pass](https://github.com/openvinotoolkit/openvino/blob/aa6d3811e6dea93cb818ff483bf6c3ca849d4034/src/plugins/intel_gpu/src/graph/include/pass_manager.h#L313), which makes different decisions for dependency settings depending on whether a node is optimized at compile time or at runtime.

2. **Runtime optimization decision**
  - Once the shape is updated in `primitive_inst::prepare_primitive()`, `do_runtime_skip_*node_type*` for each type of operation decides whether to skip the node at that execution or not.
  
3. **Caveats**
  - Once the `primitive_inst::_can_be_optimized` is set true, the runtime will only update its metadata such as shape or padding information and skip the actual execution.
  - Also, it needs to update the primitive_inst's output memory with its input memory. This is done by `update_output_memory()` called from `primitive_inst::on_execute()`.
  - If you are adding a new type of skippable operation, please make sure that the primitive has `update_output_memory()` function implemented too. 