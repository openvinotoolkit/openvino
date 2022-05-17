GPU Convolution Kernel Generator
===========================================

# Introduction

The GPU convolution kernel generator is used to generate kernels for different
kinds of convolution for Intel GPUs. The main goals for the generator are:
- Assembly level performance
- Broad coverage including data types, propagation kind, HW, etc
- Generalization
    - Rely on common building blocks and optimizations
    - Avoid specialization unless it's absolutely necessary

# Convolution

## Generalized Convolution Algorithm

See [oneDNN documentation](https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html)
for the naming conventions that are used below.

Convolution has more variations than GEMM but for simplicity we will rely on
the GEMM naming conventions to come up with a generalized convolution
algorithm. GEMM performs the following operation: `C += A * B`, where:

- `A` is `(M x K)` matrix
- `B` is `(K x N)` matrix
- `C` is `(M x N)` matrix

Algorithmically, any convolution can be expressed in a form which is very
similar to GEMM:

```python
for i0 in range(0, I0):
    ...
    for j0 in range(0, J0):
        ...
        c_val = 0
        for k0 in range(0, K0):
            ...
            a_val = load_A(i0, ..., k0)
            b_val = load_B(k0, ..., j0)
            c_val += a_val * b_val
        store_C(i0, ..., j0, ..., c_val)
```

`i0`, `j0`, and `k0` are `M`, `N` and `K` dimensions respectively:
- `M` dimensions are shared between `A` and `C`
- `N` dimensions are shared between `B` and `C`
- `K` dimensions are shared between `A` and `B`

Convolution may have many `I`/`J`/`K` dimensions.

Let's consider 1D forward convolution:

```python
for mb in range(0, MB):                                          # M
    for ow in range(0, OW):                                      # M
        for oc in range(0, OC):                                  # N
            dst_val = 0
            for ic in range(0, IC):                              # K
                for kw in range(0, KW):                          # K
                    src_val = load_src(mb, ic, ow, kw)
                    wei_val = load_wei(oc, ic, kw)
                    dst_val += src_val * wei_val
            store_dst(mb, oc, ow, dst_val)
```

Here `load_wei()` and `store_dst()` translate ND indices into 1D to calculate
the offset to load from/store to. However `load_src()` should also performs
convolution-specific `4D` to `3D` translation in the beginning. `ow` and `kw`
are translated to `iw` using the following expression `iw = ow * SW + kw * (DW + 1) - PW`.

Another convolution-specific feature is the logical padding support.
`load_src()` should return `0` in cases when `iw` index is out of bounds.

Considering all the above mentioned, load/store functions for `ncw`/`oiw`/`ncw`
layouts can be implemented as follows:

```python
def load_src(mb, ic, ow, kw):
    iw = ow * SW + kw * (DW + 1) - PW
    if iw < 0 or iw >= IW:
        return 0
    off = 0
    off += mb * IC * IW
    off += ic * IW
    off += iw
    return src[off]

def load_wei(oc, ic, kw):
    off = 0
    off += oc * IC * KW
    off += ic * KW
    off += kw
    return wei[off]

def store_dst(mb, oc, ow, val):
    off = 0
    off += mb * OC * OW
    off += oc * OW
    off += ow
    dst[off] = val
```

Backward by data and backward by weights convolutions can be expressed in the
same, GEMM-like form.

Here are the steps needed to transform any convolution to the GEMM-like
form:

- Identify M/N/K loops and map convolution tensors to A/B/C
    - For forward: source -> A, weights -> B, destination -> C
    - For backward by data: destination -> A, weights -> B, source -> C
    - For backward by weights: source -> A, destination -> B, weights -> C
- Describe load/store functions for the tensors. There are two parts:
    - Underlying layout (e.g. blocked layout `NChw32n32c`)
    - Access condition or mask (e.g. `iw >= 0 and iw < IW`). Load/store the
      element if the mask is true. Otherwise, for loads: return zero, for stores:
      drop the store.

Both steps depend on whether the convolution is forward/backward and should be
specialized accordingly. To properly do loads/stores, the generator introduces
the "view" abstraction which contains information about the tensor: its
underlying layout and corresponding masks (see the detailed description below).

## GPU Convolution Optimizations

GPU convolution requires a number of optimizations to reach close to roofline
performance:

- High-level optimizations
    - Loop order and blocking
        - Including assigning loops to the kernel grid and thread group grid
    - Single SLM buffering
        - Including selecting the SLM layout
    - Load/store decomposition into instructions
        - Block vs scattered messages
        - This also includes mask handling (for out-of-bound and stride
          conditions)
    - Multiplication decomposition into instructions (mad, dp4a, dpas(w))
        - This also includes optional GRF blocking (load -> compute -> load to
          the same GRF buffer -> compute)
        - May require auxiliary reorders to match the instruction layout
- Middle-level optimizations
    - Double/triple SLM buffering with split barriers
    - GRF buffering for SLM loads
- Low-level optimizations
    - Loop unrolling
    - Assigning GRF banks for multiplication buffers to avoid bank conflicts
    - SLM layout padding to avoid SLM write conflicts
    - Transforming dpas to dpasw to reduce SLM traffic and SLM consumption
    - Offset/address arithmetic optimizations

## Kernel Generation High-Level Flow

The generation flow consists of three main stages:

- Creating a kernel skeleton using intermediate representation (IR). The
  resulting kernel includes the high-level optimizations: loop nest,
  loads/stores, multiplications, SLM buffering.
    - After this stage the kernel is functionally correct but it needs further
      passes/optimizations to apply more low-level optimizations and to convert
      it to the form that can be lowered to assembly
- Fine-grained optimizations. This is mostly about applying low-level/local
  optimizations:
  - Transforming single SLM buffering to double/triple buffering
  - Expression simplification
  - Loop hoisting
  - Common subexpression elimination
  - Strength reduction
- Binary code generation. At this stage the kernel is fully optimized and needs
  to be translated to nGEN which is responsible for binary code generation.

# Generator Design

The main modules of the generator are:
- Intermediate representation (IR)
    - Used to describe the convolution kernel
    - The initial IR form contains all high-level optimizations (blocking,
      multiplication decomposition into instructions, etc)
    - Middle-level and low-level optimizations are implemented as IR passes
        - IR pass takes a kernel (or an IR statement) and returns a transformed
          kernel with some optimizations (e.g. with applied double SLM
          buffering)
- Expression simplification
    - Various algorithms and rules to simplify IR expressions
    - Used for offset simplification
- Binary code generator
    - Performs lowering from IR to nGEN
    - nGEN is used to generate assembly binary code
- Tensor, layout and view abstractions
    - Layout describes how logical tensor indices are stored in memory
        - Semantically it's the same as the oneDNN blocked memory descriptor
    - View describes a "virtual" tensor
        - Virtual tensor in general doesn't exist in memory, instead a view
          contains information about how to access elements of such a tensor
        - View helps to express out-of-bound/stride conditions and generalize
          forward/backward convolution algorithms
        - See the detailed description below

## IR

IR of the generator adopted many ideas from the IR used by the
[Halide](https://halide-lang.org/) project.

All IR objects are immutable by design and use reference counting. The base
class is `object_t` which implements intrusive reference-counting for
`object_impl_t` objects. `object_t` is a wrapper over the real implementation
in `object_impl_t`. All IR objects must have `object_impl_t` in their
inheritance hierarchy as the top-most class.

IR objects support equality comparison: `a.is_equal(b)`. `operator==()` is
reserved and overloaded for boolean comparisons. Additionally IR objects
provide `get_hash()` method to allow using them as keys for
`std::unordered_set` and `std::unordered_map` containers, see corresponding aliases:

- `object_map_t` - an unordered map with `object_t` as the key
- `object_set_t` - an unordered set with `object_t` as the key

Main IR objects are:

- Expressions: class `expr_t` (inherited from `object_t`). Examples:
    - Variables: `var_t`
    - Immediate values (constants): `bool_imm_t`, `int_imm_t` and `float_imm_t`
    - Unary/binary/ternary operations: `unary_op_t`, `binary_op_t`,
      `ternary_op_t`
- Statements: class `stmt_t` (inherited from `object_t`). Examples:
    - Let statement: `let_t`. Binds a value to a variable for the scope defined
      by the let statement.
    - Loop statement: `for_t`
    - If statement: `if_t`
    - Function call: `func_call_t`
- Functions: class `func_t` (inherited from `object_t`). A function and its
  arguments-expressions are used to construct a function call - statement
  with some side effects. Many GPU assembly constructs are represented with
  functions, for example:
    - Synchronization instructions: barrier-wait, barrier-signal, memory fences
    - FMA instructions: fma, dp4a, dpas(w)
    - Send instruction

IR expressions support operator overloading for convenience of use:

```c++
expr_t a = var_t::make(type_t::s32(), "a");
expr_t b = var_t::make(type_t::s32(), "b");
expr_t c = 2 * (a + b) - a;
expr_t c_simplified = simplify(c);
// (gdb) call c.dump()
// ((2 * (a + b)) - a)
// (gdb) call c_simplified.dump()
// (a + (b * 2))
```

### IR Printing and Debugging

All IR objects provide:

- Overloaded `operator<<` to use with `std::ostream`
- `str()` method returning a textual representation of the object
- `dump()` method to call it under gdb to print a textual representation of
  the object:
    - `call obj.dump()` (under gdb)

All the main IR passes trace the after-pass IR statement when tracing is
enabled (controlled by `LOG_LEVEL`).

`ir_printer_t` class is mainly responsible for the IR printing-related logic.

### Functionality to Traverse and Modify IR

A convolution kernel is an IR statement. Most IR objects contain other IR
objects. In general an IR object can be considered as a tree. Some rules apply:

- Statements can include other statements, expressions and functions
- Expressions can include other expressions but can't contain statements or
  functions

`ir_visitor_t` implements generic functionality to traverse an
arbitrary IR object. Example:

```c++
// Custom visitor to count the total number of loops in the given IR object.
class loop_visitor_t : public ir_visitor_t {
public:
    void _visit(const for_t *obj) override {
        refs++;
        // To count nested loops.
        ir_visitor_t::_visit(obj);
    }
    int refs = 0;
};

// root_stmt is an IR statement
loop_visitor_t visitor;
visitor.visit(root_stmt);
```

`ir_mutator_t` is similar to the IR visitor but is used to update IR trees.

## Expression Simplification

To be added.

## Binary Code Generator

The main logic for code generation is implemented as an IR visitor, in
`ir_to_ngen_t` class. The final IR is very close to assembly so the generation
process is straightforward. Some examples of how different IR objects are
handled:

- Let statement (to introduce variables and bind them to a value)
    - The register allocator is used to allocate a subregister for the variable
    - The variable is initialized either with a `mov` instruction or the value
      is evaluated in the subregister directly
    - Expression binding (`expr_binding_t`) is updated to bind the IR variable
      object to the subregister (to be able to access it in nested
      statements/expressions later)
    - The nested statement of the let statement is visited
    - The subregister is released after traversing the nested statement
- Expressions
    - Expression evaluation is handled by `expr_evaluator_t` class
    - Expression is evaluated recursively. For each expression:
        - Its operands are evaluated (and bound to subregisters if needed)
        - The expression itself is evaluated
    - Sometimes we want to compute an expression in a pre-allocated subregister
      (for example when lowering a let statement). This case is also supported
      by `expr_evaluator_t`.

Additionally, the generator implements extra logic for functionality such as:

- Instruction emulation. Some platforms don't have support for some instructions. Examples:
    - 64-bit arithmetic emulation. This is not handled by the generator and
      implemented in `gpu/jit/gemm/emulation.hpp`.
    - `add3` instruction. Emulated as two `add` instructions on older architectures.
- GRF region restrictions. Example:
    - `d` <-> `q` or `d` <-> `b` conversions require to align the smaller data
      type GRF region to match the other data type
- Direct implementation of IR functions. Examples:
    - Reorder between GRF layouts. For simplicity reorders are emitted in
      place. In the kernel IR they are represented as function calls.
    - Reduction of a GRF buffer. Similar to the GRF reorder.

## Tensor, Layout and View

Tensor, layout and view are the core abstractions of the generator.

**Tensor** - describes a tensor with offsets (stored as IR expressions). Example:
`32x32x1x1` tensor with `[mb_idx, ic_idx, 0, 0]` offsets (activations for 2D
convolution: `N x C x H x W`).

**Layout** - describes a memory layout, contains a physical representation of a
tensor. Layout properties:

- Data type
- Number of dimensions
- Offset to the start of the tensor (in elements of the data type)
    - Same as `offset0` in `dnnl_memory_desc_t`
- Layout blocks
    - Blocks are stored with their dimension index, block size and stride
        - Outer blocks and non-blocked dimensions are also fully specified with
          dedicated blocks
    - Example: `4n2c7h7w32n32c` (6 blocks) (`NChw32n32c` in oneDNN convention)

**View** - describes a "virtual" tensor (view) with its underlying tensor/layout:

- View tensor `V` has `m` dimensions: `V(v0, v1, ..., v(m-1))`
- Underlying tensor `T` has `n` dimensions: `T(t0, t1, ..., t(n-1))`
- Mapping from view dimensions to tensor dimensions is defined by special
  functions:
    - `t_j = F_j(v0, v1, ..., v(m-1))`
- M/N/K dimension kinds (GEMM behavior) for `V` dimensions
- Each `t_j` dimension may have an associated access mask
    - When the mask is evaluated to false, the element is assumed to be `0`

View example: 2D convolution, 3x3 filter:

- View `V` has 6 dimensions: `mb`, `ic`, `oh`, `ow`, `kh` and `kw`
- Tensor `T` has 4 dimensions: `mb`, `ic`, `ih`, `iw`
- Mapping from view to tensor:
    - `mb` is directly mapped (`t_0 = v_0`)
    - `ic` is directly mapped (`t_1 = v_1`)
    - `ih = oh * SH + kh * (DH + 1) - PH`
    - `iw = ow * SW + kw * (DW + 1) - PW`
- M/N/K dimension kinds:
    - M dimensions: `mb`, `oh`, `ow`
    - K dimensions: `ic`, `kh`, `kw`
- Access masks:
    - `mb` mask: empty
    - `ic` mask: empty
    - `ih` mask: `ih >= 0 and ih < IH`
    - `iw` mask: `iw >= 0 and iw < IW`

The view abstraction encapsulates computation semantics including
convolution-specific stride and out-of-bound conditions and M/N/K dimension
kinds. Having an outer loop nest and defined A/B/C views for the inner blocked
multiplication is enough to fully describe the convolution computation in
terms of the algorithm.

## Kernel Generation Flow

### Configuring Kernel Parameters

This is performed during primitive descriptor initialization.

Kernel configuration is defined by `config_t` object. The configuration
includes:

- Convolution problem description (`conv_problem_t`)
    - Propagation kind, batch size, input/output channels, etc
- Implementation-specific kernel parameters. Examples:
    - Block sizes
    - SLM buffering parameters: enabled/disabled, single/double/triple
    - FMA instruction to use: mad, dp4a, dpas(w)

Kernel parameters are set depending on the architecture, propagation, problem
shape, etc.

The configuration object is further passed to the kernel builder which
generates the kernel according to the configured parameters.

### Initializing Kernel ABI (kernel_arg_info_t)

This is performed during primitive descriptor initialization.

Kernel ABI defines the order of the kernel arguments, their types and their
bindings to IR expressions.

During kernel creation/generation, `kernel_arg_info_t` is used to:
1) Set up the kernel interface via nGEN
2) To access IR expressions corresponding to the kernel arguments

During execution, `kernel_arg_info_t` is used to set arguments according to the
kernel ABI.

### IR Generation 

This and further steps are performed during primitive initialization.

`kernel_builder_t` class is responsible for the whole kernel IR generation.
There are other builder classes which are responsible for more specialized
functionality, for example:

- Builder to construct load-multiply statements
- Builder to decompose view load/store into messages

#### Forming Loop Nest and A/B/C Views

The generation starts with creating the outer loop nest by forming
corresponding IR statements. The order of loops and blocking schemes are
hard-coded for different propagation kinds though some parameters are
configurable such as block sizes and the thread group size.

For simplicity there are some conventions and limitations:

- The outer loop nest is mapped to the kernel grid (grid of thread groups,
  supports up to 3 dimensions)
    - In many cases this requires packing of several problem-specific
      dimensions into one. In this case the kernel must implement unpacking
      using division and modulus operations.
- The outer loop nest typically includes M and N dimensions. It may also
  include K dimensions - in this case we have partial reduction, and the kernel
  must use atomic stores for C updates.
- Convention: thread group doesn't have outer loops across M or N dimensions,
  only across K dimensions.

According to these rules, the generic looping scheme looks as follows (assume
one dimension per M/N/K, for simplicity):

```python
for m_tg_idx in range(0, m, m_tg_blk):         # Mapped to the kernel grid
    for n_tg_idx in range(0, n, n_tg_blk):     # Mapped to the kernel grid
        for k_tg_idx in range(0, k, k_tg_blk): # Mapped to the kernel grid
            ...
            for k_idx in range(k_tg_idx, k_tg_idx + k_tg_blk, k_blk): # Loop inside thread
                ...
               # Perform C += A * B multiplication cooperatively by a thread group
               # A is (m_tg_blk x    k_blk)
               # B is (   k_blk x n_tg_blk)
               # C is (m_tg_blk x n_tg_blk)
```

After this step we have the following blocks:
- Let statements to unpack M/N/K indices from the kernel grid
- IR statements for the explicit reduction loops (inside a thread)
- A/B/C views describing thread group level blocked matrix multiplication
    - These views contain sizes, M/N/K semantics, access masks and the
      underlying layout

With these blocks the representation is generic enough so that all further
steps in the flow are common between different propagation kinds.

#### SLM Buffering, Loads, Blocked Multiplication, Epilogue and Final Store

`compute_builder_t` is responsible for generation of the innermost blocked
computation and the final store of tensor C. According to `config_t` object the
builder generates the following kernel parts:

- SLM loads and stores (when SLM buffering is enabled):
    - Define SLM layout. Use FMA-friendly layout, if reorders are necessary,
      perform them earlier, between loads from global memory and stores to SLM
    - Split a view between all threads in the thread group for cooperative loads/stores
        - Sometimes only part of the thread group should participate in
          loads/stores - use thread group sub-grid and conditions to guard loads/stores
    - Add barriers before and after stores to SLM
- Loads from SLM/global memory and multiplication
    - Split A/B thread group views across the thread group grid (X x Y x 1)
        - Split M dimension across Y dimension of the grid
        - Split N dimension across X dimension of the grid
        - Each thread computes `(m_thr x n_thr)` tile of C tensor
            - `m_thr = m_tg / Y`
            - `n_thr = n_tg / X`
    - Split per-thread blocked multiplication to sub-tiles according to the
      configuration (to reduce GRF consumption and reuse GRF buffers)
        - Typically `b_sub_tiles > 1` is used (B tile is split into sub-tiles)
    - Generate loads (from SLM or global memory) for A/B tensors
    - Generate GRF-to-GRF reorders (when needed) to match FMA layout. This is
      mainly needed for dpas.
    - Generate IR function calls matching FMA instructions
    - Apply dpas to dpasw transformation (if possible). Save GRF permutation
      `grf_permutator_t` to restore registers back after applying dpasw. This
      permutation will be applied to the final C result.
    - Restore per-thread C tensor in terms of problem layout/dimensions
        - Per-thread C tensor is `(m_thr x n_thr)` but the way M/N dimensions
          are mapped back to the problem dimensions completely depends on how A/B
          were split across the grid. `mnk_mapper_t` is used to track that.
- Epilogue (`epilogue_builder_t`) in case when convolution includes bias, post-ops or output scales
    - Bias/post-ops/output scales are handled similarly by `post_op_builder_t`
    - C is split into blocks to apply post-ops
    - Flow for applying post-ops is:
        - Pre-load the corresponding blocks of right-hand side tensors to GRF
          (for binary post-ops or for bias/output scales)
        - Convert C to `f32`
        - Generate IR statements to handle all post-ops step by step
        - Convert the updated C to the final data type
- Final stores to global memory
    - Generate stores (maybe atomic stores, for partial reduction)

#### More Optimizations and IR Passes

At this step the kernel is functionally correct. Now, more transformations and
optimizations need to be applied:

- Injecting double/triple SLM buffering
    - `simple_slm_buffering_injector_t` or `unrolled_slm_buffering_injector_t`
      is used to convert single buffering to double/triple SLM buffering
      according to some rules
- Expression simplification
- Let optimization
    - Remove unused or redundant let statements 
- Loop hoisting
- Common subexpression elimination
- Strength reduction (this is only applied with unrolled SLM buffering)
- Generating proper send headers
    - Initial `send_t` function calls contain byte offsets. Messages headers
      are generated according to the message specification.
- Peephole optimizations
    - `add` + `add` -> `add3`
    - `mul` + `add` -> `mad`

### Binary Code Generation

At this step the kernel in the IR form includes all optimizations. The binary
code generator is implemented similarly to other nGEN-based kernels. The main
differences are related to IR usage. The binary generator flow steps are
described below:

- Configuring the kernel interface:
    - Use `require\*()` API to specifying/enabling kernel features, such as:
      SIMD size, SLM size, dpas, barrier, etc
    - Listing kernel parameters according to `kernel_arg_info_t` (kernel ABI)
- Expression binding initialization for thread grid IDs and kernel arguments
    - This is to bind IR variables for external parameters the corresponding
      registers
    - Further, any reference to such an external variable is resolved based on
      the expression binding
- Visiting the kernel IR statement. The IR tree structure is recursively
  traversed and corresponding instructions are emitted using nGEN calls.

#### Lowering IR to nGEN

`ir_to_ngen_t` implements the IR visitor interface and walks through the whole
kernel. Handling of `for_t`, `if_t`, `let_t` and similar statements follows the
same pattern. First, we evaluate the related conditions, values, loop bounds.
During evaluation they are bound to some registers. After that we can form the
statement using proper instructions, e.g. `cmp`/`jmpi` for the loop or
`if`/`else`/`endif` for the if statement. The body statement is visited and
lowered to nGEN recursively.
