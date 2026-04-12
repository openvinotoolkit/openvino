# Extending `jit_kernel.hpp`

Design notes for evolving the xbyak‑based JIT helper in
`src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel.hpp`. Scope is
deliberately bounded: make writing ukernels ergonomic, optionally make the
scaffolding multi‑arch. **Fusion, scheduling and register pressure remain the
developer's responsibility.** The DSL is a typed assembler, not a compiler.

MLIR and similar compiler‑backed approaches are explicitly out of scope here —
see the end of the document for why.

## What the current wrapper already does well

`jit_kernel` layers on top of `dnnl::impl::cpu::x64::jit_generator_t` and gives:

- **Value semantics over registers.** `variable<T>` / `variable<T[N]>` wrap a
  `shared_reg` whose deleter calls `kernel.free()` (`jit_kernel.hpp:881`).
  Lifetimes follow C++ scopes — no manual reg tracking.
- **Control flow as callbacks.** `foreach`, `_if(...)._then([]{...})._else([]{...})`
  (`jit_kernel.hpp:817`, `jit_kernel.hpp:198`).
- **Scalar operator overloads.** `+ - * & | << >>` on `variable<T>`
  (`jit_kernel.hpp:327‑446`).
- **Emitter interop, literal pool, stack frames** (`jit_kernel.hpp:555`, `:577`,
  `:752`).

The gap that prevents kernels from reading as expressions rather than as
`uni_vmulps`/`uni_vaddps` sequences is the explicit `// TODO: implement vector
arithmetic` at `jit_kernel.hpp:552`, plus the absence of a composition mechanism
for stages.

## Phase 1 — Vector arithmetic operators (prerequisite)

Non‑negotiable. Without this, nothing else pays off; with this alone, the
existing imperative code at `color_convert.cpp:233‑266` collapses from ~20 lines
of `uni_v*` calls into ~4 lines of expression.

**Rule: one overload = one instruction, or it doesn't exist.**

This rule is the whole design. It's what keeps the layer a typed assembler
rather than drifting into a smart‑codegen thing that fights the developer.

In scope:
- `operator+ - *` on `variable<T[N]>` → `uni_vaddps` / `uni_vsubps` / `uni_vmulps`
- `operator&` `|` `^` on integer vector variables → `uni_vpand` / `uni_vpor` /
  `uni_vpxor`
- `max`, `min`, `abs`, `sqrt`, `rcp`, `rsqrt`, `round` as free functions
- `fma(a, b, c)` as a free function — **not** `a*b + c`

Not in scope:
- `operator/` on vectors. Division is either `rcp + Newton` or a library call;
  the developer must pick. Hiding that choice violates the one‑instruction rule.
- Implicit scalar→vector broadcast. Use `broadcast(k)` explicitly.
- Any overload whose lowering changes based on surrounding liveness (e.g.
  deciding between `vfmadd213/231/132` automatically). That's exactly the kind
  of hidden decision the developer said they want to control.
- `operator+=`, `-=`, `*=` on vectors. See Phase 3 note on 3‑operand form — keep
  vectors 3‑address from day one so the arch generalization doesn't have to
  undo this later.

Acceptance test: rewrite the math block in
`jit_uni_converter::yuv_to_rgb` (`color_convert.cpp:247‑260`) using the new
operators. If the result isn't clearly more readable than the current form,
stop and reconsider before doing any of Phase 2.

## Phase 2 — Stages, pipes, tiles

Only do this once Phase 1 has shipped and there is a concrete second caller.
**One caller does not justify a DSL.** Color_convert alone is not enough.

### `tile<Ts...>`

A `std::tuple<variable<...>>`. Pure carrier, no behavior. ~20 lines. Exists
solely so a stage can return `{r, g, b}` and the next stage can destructure.
Without it, multi‑output stages have to take out‑params by reference and the
"chain" reads worse than the imperative version.

### `Stage` concept

A stage is any callable `Out(jit_kernel&, In...)`. No base class, no virtual,
no type erasure — just a concept constraint. Use C++20 `concept` to get
tolerable error messages; without concepts the template errors are
Eigen‑class.

### `operator|` composition

```cpp
template <class F, class G>
auto operator|(F f, G g) {
    return [f, g](jit_kernel& k, auto&&... xs) {
        return std::apply(
            [&](auto&&... ys) { return g(k, decltype(ys)(ys)...); },
            as_tuple(f(k, decltype(xs)(xs)...)));
    };
}
```

~30 lines including the `as_tuple` shim that handles both single‑value and
`tile` returns. **It allocates nothing, emits nothing, schedules nothing.** It
literally calls `g(k, f(k, ...))`. Every instruction still comes from the
stage bodies — the composition is pure syntactic sugar for nesting.

This is important: the composition mechanism does not introduce hidden spills,
hidden loads, or cross‑stage register reuse. Register pressure crossing a
stage boundary is exactly the same as if the developer had inlined the
callees. That's a *feature* — predictability is the point.

### Scoped scratch pool helper

```cpp
auto scratch = k.scratch<float[N], 3>();   // reserves t0,t1,t2; freed on scope exit
auto [t0, t1, t2] = scratch;
```

RAII already frees per‑variable; this is just the ergonomic form so a stage
declares "I need exactly 3 scratch vectors" up front. Makes register cost of
a stage visible at a glance, which is critical when the developer is doing
fusion reasoning by hand.

### Free‑function `load_*` / `store_*` / `broadcast_*` as stages

So a kernel body becomes:

```cpp
foreach_(0, width,
    load_yuv(src_y, src_uv)
  | yuv_to_rgb(_consts, color_format)
  | store_rgb(dst));
```

Compare to the current `JitConverter<T[N]>::generate()` in
`color_convert.cpp:432`, which manually interleaves load, body, store, plus a
separate tail path at `color_convert.cpp:472`.

### Explicit non‑features

- **No auto‑spill.** If a stage needs more registers than available, `reserve<>`
  fails. That is the correct signal; the developer must refactor. Auto‑spill
  would hide exactly the cost the developer needs to see.
- **No deferred IR, no lazy emission.** Stages emit top to bottom at call
  time, same as today. Predictability > flexibility.
- **No implicit loop fusion between chained `foreach`es.** `foreach` is
  terminal. To fuse two passes, write one body.
- **No "smart" instruction selection.** `operator+` emits `vaddps`, period.
- **No ISA polymorphism inside a stage.** Stages are templated on `T[N]` like
  `JitConverter<T[N]>` is today (`color_convert.cpp:425`).

## Phase 3 — Arch generalization (only if multiple arches need it)

**Trigger condition: at least 3 kernels that want the same high‑level shape on
multiple arches.** The evidence already in‑tree is
`jit_uni_eltwise_generic.{cpp,hpp}` existing in `kernels/x64/`,
`kernels/aarch64/`, and `kernels/riscv64/` as three parallel copies. Before
committing to this phase, diff those three files. If they really are
structurally the same, a shared base pays for itself. If they've diverged
materially, the abstraction will leak.

### What's arch‑invariant (lifts cleanly)

- `consts_table` (`jit_kernel.hpp:577`) — literal pool, no instructions.
- `shared_reg` + RAII register pool (`jit_kernel.hpp:881`) — lifetime bookkeeping
  over a free list.
- `stack_frame` (`jit_kernel.hpp:555`) — shape is `sub sp, K; ...; add sp, K`
  everywhere.
- `foreach`, `_if/_then/_else` — the *shape* `cmp; jcc; body; jmp back; label`
  is identical on x86 / aarch64 / RVV. Only mnemonics differ.
- `stage` / `pipe` / `tile` — no instructions at all.

### What's arch‑specific (does not lift)

- `reg_traits_by_size` (`jit_kernel.hpp:40‑86`). The "sizeof(T) →
  Reg8/16/32/64/Xmm/Ymm/Zmm" mapping is pure x86. aarch64 has
  `XReg/WReg/VReg/ZReg/PReg`; RVV has GPR + V with `LMUL`.
- `isa_traits` with `length = 4/8/16 dwords` (`jit_kernel.hpp:117‑141`) —
  assumes *fixed* vector length. SVE and RVV are scalable.
- 2‑operand vs 3‑operand form. `operator+=` compiles to `add dst, src` on x86;
  ARM and RVV are 3‑operand. This is why Phase 1 rules out vector `+=`.
- Predication. SVE and RVV are predicated first‑class; AVX‑512 has k‑regs;
  AVX2/SSE has none.
- `jit_load_emitter` / `jit_store_emitter` (`jit_kernel.hpp:752`) — x86‑only in
  oneDNN.

### The boundary

```
jit_kernel_base<ArchTraits>              // shared scaffolding
  ├─ ArchTraits_x64                       // reg types, vec ops, control flow
  ├─ ArchTraits_aarch64
  └─ ArchTraits_riscv64

jit_kernel_x64     : jit_kernel_base<ArchTraits_x64>,     dnnl::x64::jit_generator_t
jit_kernel_aarch64 : jit_kernel_base<ArchTraits_aarch64>, dnnl::aarch64::jit_generator_t
jit_kernel_riscv64 : jit_kernel_base<ArchTraits_riscv64>, riscv64::jit_generator
```

`ArchTraits` declares a small fixed vocabulary — ~20 ops — that every arch
implements:

```
load broadcast store
add sub mul fma
max min abs round sqrt rcp rsqrt
cmp_lt cmp_le cmp_eq blend
cvt_f32_i32 cvt_i32_f32
```

That's the Eigen `packetmath.h` pattern. It works because every op has a
single obvious lowering on every arch, and because you refuse to add anything
that doesn't.

### Hard compromises — accept them up front

These are non‑negotiable for the abstraction to remain sane. If you can't live
with any of them, don't start Phase 3.

1. **Fixed vector width.** `variable<T[N]>` keeps compile‑time `N`. On aarch64
   target NEON (`N=4` for `float`). On RVV pin `LMUL=1` and emit one
   `vsetvli` at the top. Scalable SVE/RVV is **not** supported through this
   DSL — if you need scalable form, write a separate source file. Fighting
   this rule produces a DSL where no one can reason about register count or
   live range, and drags the x86 experience down for no benefit.
2. **3‑operand form in the DSL, lowered to 2‑op on x86.** No vector `+=`.
   `uni_*` already does the 2‑op lowering silently on x86. Cost: slightly
   more verbose x86 source. Benefit: aarch64 and RVV are not second‑class.
3. **Predication as an optional last argument**, defaulted to all‑lanes. On
   AVX2 passing non‑default is an error. On AVX‑512 it becomes a `k` mask.
   On SVE/RVV it becomes a `P`/`v0.t` mask. Tails that can't use predication
   fall back to a scalar loop written in the same DSL.
4. **Arch‑specific extensions live in arch‑specific files**, not behind
   `#ifdef`s in shared code. A kernel that needs `vpshufb` has
   `yuv_to_rgb_x64.cpp`, and if/when someone cares, `yuv_to_rgb_aarch64.cpp`.
   Don't fake cross‑arch unity the DSL can't actually deliver.

### What does not get unified

- `src/plugins/intel_cpu/thirdparty/onednn/src/cpu/aarch64/jit_generator.hpp`
  (770 lines) is oneDNN upstream, uses `Xbyak_aarch64` (not xbyak), uses
  `uni_fadd` where x86 uses `uni_vaddps`. The misalignment is at the
  `jit_generator` layer, with different maintainers and conventions.
  **Do not try to unify it.** Layer on top of both `jit_generator` variants,
  exactly as the current x86 `jit_kernel` layers on top of
  `dnnl::x64::jit_generator_t` (`jit_kernel.hpp:594`).

## API alignment with SIMD wrappers

The phases above were written before a second reference point existed.
Since then, two have, and both sit on the AOT side of the JIT/AOT split:

- `std::basic_simd` (C++26). The standard library's portable SIMD value
  type. Rides on the host compiler's regalloc, scheduling, and instruction
  selection.
- `simd_avx2.hpp` et al. at
  `../turbo_quant/src/plugins/intel_cpu/src/nodes/kernels/simd/simd_avx2.hpp`.
  An in-tree AOT intrinsics wrapper with a `vec<T, isa>` value type, `+ - *`
  operator overloads, free-function `load` / `store` / `reduce` / `fmadd`,
  and comparison operators returning a `mask<isa>`.

Both wrappers own regalloc, spill, and instruction selection through the
C++ compiler. `jit_kernel` owns all three manually, at emit time. That gap
is load-bearing: it forces some API differences that will never go away.
But many of the current differences between `jit_kernel` and these wrappers
are *gratuitous* — same concept, different spelling — and closing them lets
kernel bodies read the same across backends.

**Goal.** A kernel body written against the `jit_kernel` value API should
read as close as possible to the same kernel written against
`std::basic_simd` or `simd_avx2.hpp`. Same names, same argument orders,
same return-by-value conventions. A developer moving between AOT and JIT
backends should be relearning *constraints*, not *vocabulary*.

**Hard filter.** Every candidate match must still pass the Phase 1 rule:
**one overload = one instruction, or it doesn't exist**. A renamed free
function that emits one instruction is a clean win. A constructor that
quietly emits three instructions because it matched a mask-based overload
is exactly the kind of hidden decision the rule exists to prevent.

### Translate AOT constraints, don't copy them

A second failure mode when aligning with `simd_avx2.hpp` or
`std::basic_simd`: mechanically copying API shapes whose *existence* is
forced by a constraint that applies only to the AOT side. Template
parameters on immediate arguments are the canonical instance.

**Example — the `shuffle` shape.** `simd_avx2.hpp` declares:

```cpp
template <int imm>
inline vec<float, isa::avx2> shuffle(vec<float, isa::avx2> a, vec<float, isa::avx2> b);
```

The template parameter here is not a stylistic choice. `_mm256_shuffle_ps`
requires `imm` to be a C++ *compile-time* constant — the compiler lowers
it directly into the instruction byte during AOT compilation and rejects
any other form. If you want an imm8-shuffle intrinsic wrapper, a template
parameter is the *only* way to express the argument.

`jit_kernel` has no such constraint. `uni_vshufps(dst, src1, src2, imm)`
is a C++ method that writes bytes into a code buffer at **emit time**,
which is C++ runtime. Its `imm` is a plain `uint8_t` that can come from
anywhere — a literal, a `jcp_` field, a runtime `cpu_isa_t` dispatch.
Nothing in the lowering path requires it to be a C++-compile-time
constant.

Copying the template form into jit_kernel (as the first cut of
`shuffle<imm>()` did) imports two costs with no corresponding benefit:

1. **The `.template` disambiguator in dependent contexts.** Inside any
   templated kernel body (e.g. `JitConverter<T[N]>::unpack_uv`), C++
   grammar requires `uv.template shuffle<even_mask>()` — pure
   boilerplate the runtime-mask form doesn't need.
2. **Loss of emit-time runtime flexibility.** A kernel that wants to
   pick the shuffle pattern from observed layout —
   `uint8_t m = jcp_.interleave ? 0xA0 : 0xB0; auto u = uv.shuffle(m);`
   — just works with a runtime argument. With a template argument you
   need `if constexpr` branches or `_if/_then/_else` splits that
   duplicate the shuffle body.

**The rule.** When a candidate feature on the AOT side uses a template
parameter, ask *what constraint forced it* before copying the shape.

| AOT constraint | Example | JIT equivalent |
|---|---|---|
| Intrinsic semantics require `constexpr` arg | `_mm_shuffle_ps(a, b, imm)`, `_mm_extract_epi32(v, lane)` | **Plain runtime argument.** The JIT's emit-time lowering replaces the AOT compiler's compile-time lowering. |
| Template carries type identity | `static_simd_cast<DstSimd>(src)`, `vec<T, isa::avx2>` ISA tag | **Keep the template parameter** — type information still needs to flow through the C++ type system even in the JIT case. |
| Template carries dimension / N | `fixed_size_simd<T, N>` | **Keep the template parameter** — dimensions affect register choice and must be known to the type system. |
| Style (no constraint forced it) | rare, but exists | **Runtime argument.** Don't inherit a constraint nobody imposed. |

`shuffle` was an instance of row 1 misdiagnosed as row 4 — the template
was copied because `simd_avx2.hpp` had it, without asking *why*. The
corrected form takes the imm8 as a runtime argument, matching how
`permute(order)` already works: both take their selector as runtime data.

### Value-transforming operations are by value

Every operation on `variable<T[N]>` that *transforms* the contents of a
vector — arithmetic, shuffle, permute, cast — returns a fresh
`variable<T[N]>` rather than mutating `*this`. The source is untouched;
the caller decides what to do with the result.

```cpp
auto r_perm = r.permute(order);     // r preserved
auto sum    = a + b;                // a, b preserved
auto u      = uv.shuffle(mask);     // uv preserved — same call can be reused
```

This is non-negotiable for new operations. Four independent arguments
converge on the same answer:

1. **`std::simd` (C++26, P2664).** The permutation API proposal and the
   rest of `std::datapar` uniformly return by value. No in-place permute
   exists in the standard; arithmetic operators follow suit. The committee
   chose value-type cleanliness over any ISA-specific micro-optimization.
2. **`simd_avx2.hpp`.** The in-tree AOT wrapper is by-value throughout —
   `operator+ - *` return new vecs, `shuffle<imm>(a, b)` and
   `unpack_lo(a, b)` are free functions returning new vecs, no member
   mutates. We've been aligning with this wrapper for other reasons; this
   is one more.
3. **The Phase 3 3-operand rule** (already above in this doc, under "Hard
   compromises"): *"3-operand form in the DSL, lowered to 2-op on x86."*
   The DSL-level API is non-destructive by design; destructive-form
   lowering is a per-ISA backend concern. In-place member methods directly
   contradict this rule.
4. **Hardware reality on every non-x86 architecture.** ARM NEON
   (`TBL`, `EXT`, `ZIP1/2`, `UZP1/2`, `TRN1/2`), ARM SVE (`TBL`, `ZIP`,
   `UZP`, `SPLICE`), RVV (`vrgather.vv`, `vslideup/down`) are all 3-operand
   non-destructive. There is no "in-place permute" encoding on any of them
   — RVV's `vrgather` even architecturally *requires* `vd != vs1` to avoid
   read-after-write hazards. The only architecture where "in-place permute"
   is representable at the instruction level is x86 SSE, and oneDNN's
   `uni_v*` wrappers already abstract over it. Keeping an in-place form in
   the DSL is modeling a single legacy instruction encoding at the expense
   of every modern ISA.

**Cost acknowledgment.** On modern x86 with move-elimination, the
in-place-plus-self-assign idiom `r = r.permute(order);` produced
`vpermps r,r,order + vmovups r,r` where the `vmovups` was eliminated at
rename stage (0 ALU uops). The by-value form emits `vpermps tmp,r,order +
vmovups r,tmp` — one real `vmovups` that is *not* eliminated (distinct
registers). Net cost: one extra ALU uop per rebinding call site. For
color_convert's 5 permute call sites on a 1080p frame, that's well under
10 microseconds — invisible against the kernel's actual runtime. On
ARM/SVE/RVV the by-value form is what the instruction set emits anyway,
so there's literally no cost.

**Exceptions.**

- **Construction + load.** `k.var<float[N]>(src_ptr)` is construction from
  data, not a value transformation. It reserves a register *and* fills it.
  This is factory-shaped for register-tracking reasons (see "Principled
  divergences" below), not by-value-shaped, and the distinction is
  intentional.
- **`operator=` on `variable<T[N]>`.** Assignment is where data actually
  transfers. It emits a `uni_vmovups` to copy contents. This is the one
  place where a method *mutates* `*this`, and it's explicit at the call
  site (`=` sign) rather than hidden inside a named method.
- **Control-flow constructs** (`_if/_then/_else`, `foreach`). Not
  value-transforming — they plant labels and branches.

Everything else returns by value. When adding a new method, check this
rule before writing `const variable& foo(...) const { ... return *this; }`.

### Already matched or in progress

| Feature | jit_kernel form | Status |
|---|---|---|
| `operator+ - *` on vector `variable<float[N]>` | `a + b`, `a - b`, `a * b` | Phase 1 — shipped |
| Load-and-construct in one step | `k.var<float[N]>(src_ptr)` | shipped (post-Phase-1 follow-up) |
| Load-and-construct with runtime length | `k.var<float[N]>(src_ptr, width)` | shipped — single-expression form, no `where(mask).copy_from` dance |
| Load-and-construct with compile-time length | `k.var<float[N]>(src_ptr, N/2)` | shipped |
| In-lane shuffle with imm8 selector | `v.shuffle(imm)`, `v.shuffle(other, imm)` — runtime arg, not template | shipped — matches `simd_avx2.hpp`'s `shuffle<imm>(a,b)` semantically, but with imm as a runtime `uint8_t` (see "Translate AOT constraints, don't copy them" above for why) |
| Lane permutation (`vpermps`) | `v.permute(order)` returns fresh variable | shipped — converted from in-place mutation to by-value (see "Value-transforming operations are by value" above for the four arguments forcing this shape) |

### Match when a caller appears (not before)

All of these are low-cost additions individually, but building them
speculatively violates the "one caller does not justify a feature"
principle at the top of this document. Add each one the first time a kernel
body in this plugin becomes clearly more readable with it in hand.

| Feature | basic_simd form | Proposed jit_kernel form | Trigger |
|---|---|---|---|
| Rename width member | `basic_simd::size()` | `variable<T[N]>::width` (currently `::length`) | First kernel that reads the member directly; today no caller does |
| Element type alias | `basic_simd::value_type` | `variable<T[N]>::value_type` | Same — no caller yet |
| Broadcast factory | `basic_simd<float> v(1.0f)` | `k.var<float[N]>(1.0f)` returning a register broadcasting the scalar | `yuv_to_rgb` has a local `bc()` helper; second caller justifies promoting it |
| Alignment hint | `basic_simd<float> v(ptr, vector_aligned)` | `k.var<float[N]>(src_ptr, aligned)` — add `aligned_tag` | Caller that both knows its pointer alignment and can measure the perf delta |
| Generator factory | `basic_simd<float>([](auto i){ return f(i); })` | `k.var<float[N]>([](int i){ return f(i); })` — bakes the resulting vector into `consts_table` | Caller constructing a runtime-dispatched constant table |
| `fmadd` free function | `std::simd::fma(a, b, c)` | `fmadd(a, b, c)` returning a fresh `variable<float[N]>` | Kernel with an actual accumulator pattern (`acc = fmadd(x, y, acc)`). Color_convert is not this kernel — its arithmetic has no accumulators, so `a*b + c` and `fmadd(a,b,c)` emit the same number of instructions. |
| Bitwise `& \| ^` | `a & b` on integer vectors | Free operators on `variable<intN[M]>` | Already listed in original Phase 1 scope; still no caller |
| `max`, `min`, `abs`, `sqrt`, `round` | Free functions | Free functions taking/returning `variable<float[N]>` | Phase 1 lists them; no caller yet |
| `reduce`, `hmin`, `hmax` | Free functions | Same, returning `variable<T>` scalar | Kernel with a horizontal reduction |
| Mask type + vector comparisons + `select` | `basic_simd_mask<T>`, `a < b`, `select(m, a, b)` | `mask<N>` wrapping `shared_reg` of k-reg (AVX-512) or vector reg (AVX2); vector comparison operators returning it; `select` free function | Kernel with lane-wise masking that isn't expressible via immediate-mask `blend`. Color_convert uses immediate-mask blends only — it is not this caller. **When this lands, land the whole cluster together**: mask type, vector comparisons, `select`, masked stores, `reduce`. Do not drip the pieces in separately. |

### Principled divergences — document and stop re-litigating

These are not gaps to close. They are forced by the JIT/AOT split or by the
register-tracking requirement, and the design should explicitly keep them.
Naming them here saves the same back-and-forth in every future review.

| Divergence | basic_simd | jit_kernel | Why |
|---|---|---|---|
| Construction entry point | Constructor (value type) | Factory method on `jit_kernel` | `variable<T>` is a refcounted handle to a register pulled from the kernel's free list. Direct construction would need the kernel reference at every call site; the factory threads it implicitly through `this`. Same pattern and reasoning as `shared_ptr`/`make_shared`, but with a much scarcer resource (16 ymms vs. gigabytes of heap). |
| Default construction | Zero-initialized | Register reserved, contents unspecified | Emitting `vpxor` per default-constructed variable costs an instruction the caller usually doesn't want (they're about to overwrite it). JIT declines to pay for it. |
| Input iterator concept | `contiguous_iterator` | `variable<const T*>` | Iterator concept is C++ compile-time machinery. Our pointers exist at emit time as register-resident runtime values. The two cannot converge. |
| Division operator | `a / b` returns simd | Declined | Division is multi-instruction (`rcp + newton` or library call). Violates the one-overload-one-instruction rule. Developer spells out the approximation or trap explicitly. |
| `where(mask, v).copy_from(ptr)` for partial loads | First-class in basic_simd | Not matched — use `var<T>(ptr, width)` directly | basic_simd has no partial-load constructor because the standard preferred type-system cleanliness (all lanes either loaded or zero, never "undefined beyond k"). We are not bound by that constraint; the single-expression form is better for the JIT usage pattern. |
| `copy_from` / `copy_to` methods on the vector | First-class | Free-function `load(dst, src)` / `store(dst, src)` | Adding methods is pure syntactic sugar over the existing free-function forms. If a future design wants both, land them together; never one side only. |

### Places we deliberately do better than basic_simd

Worth naming explicitly, because framing convergence as "jit_kernel catches
up to basic_simd" will accidentally sand off real advantages.

- **Single-expression runtime partial load.** `k.var<float[N]>(src, width)`
  is one expression. basic_simd's equivalent is three statements:
  zero-init, build mask from a lane-index predicate, masked `copy_from`.
  Our form is directly expressive of the tail-handling pattern kernels
  actually use.
- **Type-converting load as a first-class primitive.**
  `jit_load_emitter` dispatches on `(src_prc, dst_prc, length)` and emits
  the fused widening/narrowing sequence (u8→f32, bf16→f32, f16→f32, etc.)
  internally. basic_simd needs `static_simd_cast<simd<DstT>>(simd<SrcT>(ptr,
  flags))`, and whether the compiler fuses load + cast is
  quality-of-implementation, not guaranteed by the spec.
- **Baked runtime constants.** A JIT kernel can stash a literal into
  `consts_table` after having observed the tensor's actual shape or scale,
  then reference it as an immediate. basic_simd has no equivalent — every
  constant must be known at C++ compile time.

### Orthogonal perf gap — not an API alignment change

The runtime-length `load` / `store` implementations in
`jit_kernel.hpp:793-813` and `:840-852` go through a stack-bounce
trampoline: reserve stack, zero-fill, scalar `foreach` loop, full-vector
reload. Correct everywhere but leaves real perf on the table on AVX2
(`vmaskmovps` / `vpmaskmovd`) and AVX-512 (`vmovups zmm{k}`), which is
exactly what `simd_avx2.hpp`'s `partial_load` emits in one instruction.

This is **not** an API alignment change and should not be bundled with one.
It's a perf upgrade inside `jit_load_emitter` / `jit_store_emitter`. Before
doing it:

- **Pick a kernel whose tail is measurably hot** on a real workload. MVN on
  small reduction axes, reduce, interpolate are candidates; color_convert
  is not (tail is ~0.4% of a 1920-pixel row).
- **Mind the AVX2 masked-store corner.** Writes to masked-out lanes still
  touch memory protection flags; this has bitten other projects. A
  defensible asymmetric shape is "maskload for loads on AVX2+AVX-512,
  scalar-tail fallback for stores on AVX2, maskstore for stores on
  AVX-512".

## Register allocation and pressure analysis

The by-value refactor (value-transforming ops return fresh registers, move
constructor and `operator=(variable&&)` rebind the handle without emitting
`vmovups`) makes the DSL more composable but shifts peak register pressure
upward relative to the old fluent/in-place style. The color_convert
investigation on 2026-04-11 quantified this gap empirically and showed it
blocking aggressive hoisting in a real kernel — see "Empirical evidence"
below. The current shared_reg pool is a scope-based refcount: registers are
released when the C++ `variable` goes out of scope, not at the last actual
use of the value they hold. For simple kernels this is close enough to
optimal; for kernels with loop-invariant values that could be hoisted, or
long-lived-but-dead intermediates that could be reclaimed, the gap between
"what a register allocator would do" and "what the DSL currently does" is
large enough to observe in emitted code.

**Decision**: implement a lightweight two-pass register allocator as an
opt-in IR mode (step 4 below). The analysis of tiers and tradeoffs below
remains useful documentation for *why* we took this path, but the
recommended ordering commits to building it rather than gating on further
measurement.

### The pressure gap, concretely

Old fluent style:
```
result.blend(g, m0);           // in-place, no new register
result.blend(b, m1);            // in-place, no new register
```

New by-value style:
```
auto t1 = result.blend(g, m0);  // fresh register allocated for t1
auto t2 = t1.blend(b, m1);      // fresh register for t2, t1 dies after
```

With move-assignment (shipped): `result = result.blend(g, m0).blend(b, m1)`
costs one transient extra register during the expression window but lands at
the same steady-state pressure as the old form.

Without move-assignment: every value-producing op allocates a fresh register
AND the assignment copies it into the LHS via `vmovups`, leaving the
temporary alive until end-of-expression. Peak pressure grows with chain
depth.

For `yuv_to_rgb` on AVX2 the gap is ~1-2 registers, well inside the 16-YMM
budget. For a kernel already sitting at 14-15 peak (e.g. MVN's reduction
paths), the by-value refactor could tip it into spills. **Without
measurement we're guessing.**

### What's observable today (zero code changes)

- `free_x64regs()` / `free_rmmregs()` accessors on `jit_kernel` already
  expose the live free list. `16 - free_rmmregs().size()` gives the
  currently-allocated count on AVX2 at any point during emission.
- `shared_reg::use_count()` via `variable_base::shreg()` tells you how many
  references exist to a given register. Does **not** tell you "about to die"
  (see "The lookahead problem" below) but useful for spot checks.

### Cheap tier — peak tracker and allocation stats

**~30 minutes of work, ~50 lines of code, guarded under
`ENABLE_DEBUG_CAPS`.** Add two counters to `jit_kernel` and instrument
`reserve<Vmm>()` / `free(Vmm)`:

```cpp
struct register_stats {
    size_t allocations = 0;     // total var<T[N]>() calls
    size_t peak_live = 0;       // max simultaneous live vector registers
    size_t pool_size = 0;       // total vector registers in the pool
};

const register_stats& vec_register_stats() const;
```

Print at `postamble()` or expose for test consumption. Output looks like:
```
yuv_to_rgb<8>: allocations=23, peak_live=11/16, pool_utilization=69%
```

One line per kernel. Tells you: is this kernel safe under the refactor, or
close to the limit. Zero runtime overhead in production builds — the
counters compile out under `#ifndef ENABLE_DEBUG_CAPS`.

**Do this first.** Everything below depends on having this visibility to
decide whether deeper work is justified.

### Medium tier — allocation log with lifetime intervals

~150 lines, builds on the cheap tier. Record every `reserve` / `free` with:
- A monotonic "op index" (bump per xbyak emit)
- Which physical register was picked
- Optional caller annotation passed to `var<T>()`

Dump as CSV:
```
alloc_op, free_op, register, lifetime_ops
17,       32,      ymm5,     15
18,       22,      ymm6,     4
```

From this you can compute:
- **Average register lifetime** (long-lived = held longer than needed?)
- **Reuse rate** (how many distinct variables shared the same physical slot)
- **Pressure curve** (live count per op index — peaks show where the
  kernel is tightest)
- **Fragmentation** (was there headroom when allocations failed?)

Useful for diagnosing why a specific kernel is tight. Not worth the cost
until the cheap tier shows a kernel actually at risk.

### The lookahead problem

All the cheap/medium observations are passive — they measure what happened.
Active optimization (picking better registers, reusing dying slots) needs to
know what **will** happen. `shared_reg::use_count() == 1` means "uniquely
owned right now", not "will not be used again". A variable can be uniquely
owned and still have many future uses. Distinguishing the two requires
lookahead.

Three approaches to get lookahead, from cheapest to most complete:

**1. Move-hint at call sites (caller annotation)**. Caller writes
`std::move(a)` to promise "this is the last use". DSL adds rvalue-ref
overloads for value-producing ops that can reuse the moved-from register
instead of allocating a fresh one:

```cpp
variable<float[N]> fma(variable<float[N]>&& a,
                       const variable<float[N]>& b,
                       const variable<float[N]>& c);
```

Inside this overload the register holding `a` is known to be the caller's
last reference, so the result can steal it. No fresh allocation, no seed
move.

- Cost: ~150 lines (overloads for every value-producing op).
- Downside: verbose (`r = fma(std::move(v), u, y)`), forgettable, no help
  across function boundaries.
- Upside: zero infrastructure, works with the eager emission model as-is.

**2. Expression templates / lazy emission**. `fma(a, b, c)` returns a
lightweight node instead of emitting immediately. `operator=` triggers the
actual emit with knowledge of the enclosing statement. The emitter sees
"this temporary is bound to `r`" and can target `r`'s register directly,
eliding the fresh allocation and the seed move.

- Cost: ~500 lines, touches every operator and every op that returns a
  variable.
- Downside: partial lookahead — only within a single statement. Deeper
  patterns (across statements, across branches) still need the big hammer.
- Upside: no control-flow complication, eager emission preserved at the
  statement boundary.

**3. Two-pass IR with linear-scan allocation — scoped narrowly**. Pass 1
records ops into an IR without emitting. Pass 2 runs linear-scan register
allocation over the IR and emits xbyak with the allocated registers.

The scope matters, and it's much narrower than "build a compiler":

- **IR is a register allocator, not an optimizer.** Explicit non-goal:
  optimization passes. No CSE, no constant folding, no LICM, no dead-store
  elimination, no pattern-matching `mul+add → fma`. The author writes
  `fma(a, b, c)` explicitly when they want FMA; the IR does not contract
  expressions the author chose to keep separate.
- **Phase 1 rule is preserved.** One DSL call = one instruction, recorded
  as such in the IR and emitted as such after allocation. Users still
  predict the emitted instruction sequence from the source. The only thing
  the IR changes is *which physical registers* the instructions reference.
- **The author controls instruction selection.** The long tail of "which
  FMA variant should I use here" or "vblendvps vs cmp+vblendps" stays in
  the author's hands via explicit primitive names. The IR does not re-open
  instruction-selection decisions the author already made.

Under this framing, the IR exists for exactly one reason: a kernel author
cannot easily reason about global register lifetimes while writing locally
readable code. Everything else (what instructions, what order, what
scheduling) is the author's responsibility. This matches the project's
design philosophy and dramatically simplifies the scope.

**Cost (scoped)**: ~900 lines total, spread across:
- Linear-scan allocator: ~150 lines (~100 lines of classical Poletto &
  Sarkar 1999, plus interval bookkeeping).
- IR data structures: ~200 lines (op list with abstract value ids, no
  type system, no optimizer-friendly canonicalization).
- Lowering from IR to xbyak: ~200 lines (pure mechanical substitution of
  allocated registers into pre-recorded ops).
- Control flow handling (`_if`/`_then`/`_else`/`foreach`): ~200 lines
  (CFG linearization, interval splitting across branches). Unavoidable
  for correctness — this is the one part that stays non-trivial.
- Debug tooling to inspect the IR: ~100 lines.

Notably absent from the list: the "~500 lines across dnnl integration" for
emitter composition. That chunk goes away under the load/store split
described in the next subsection.

- **Downside**: still a pipeline change from eager to two-pass within IR
  mode. Debugging a kernel in IR mode requires reading the IR dump, not
  the xbyak output at emit time. Roughly one week of focused work.
- **Upside**: optimal allocation within the IR-mode regions, visible
  pressure curves, no user-visible API changes for kernels that adopt it.

### IR-mode kernels and the load/store split

The biggest friction in the earlier IR estimate was emitter composition:
`jit_load_emitter` / `jit_store_emitter` / activation injectors emit xbyak
eagerly, and a lazy jit_kernel can't compose with them without either
flushing at every emitter call (defeats the purpose) or making the
emitters lazy (big cross-project refactor). This friction dissolves once
you observe that **most of what the emitters do is a type-conversion
switch table, not something "special" that requires centralization**.

`jit_load_emitter` + `jit_store_emitter` are ~1435 lines across load and
store combined. That code centralizes:

- **Type conversion**: `u8 → f32`, `bf16 → f32`, `f16 → f32`, `f32 → u8`
  with saturation, `f32 → bf16` with rounding, and every other precision
  pair. Most pairs are short instruction sequences (`vpmovzxbd +
  vcvtdq2ps` for u8→f32, `vcvtps2dq + vpackssdw + vpackuswb` for f32→u8
  with saturation). Some pairs are genuinely hard: BF16 and F16
  conversions require the `jit_uni_vcvtneps2bf16` helper and special
  rounding handling, and F16 needs the F16C extension.
- **Partial loads with fill values**: masked load on AVX-512, stack-bounce
  or `vpmaskmovd` fallback on AVX2, plus identity-value fills (`zero`,
  `float_min`, etc.) for reductions.
- **ISA dispatch**: different instruction choices across SSE4.1 / AVX /
  AVX2 / AVX-512F / AVX-512BW / AVX-512VL.
- **Corner cases**: SSE's implicit `Xmm(0)` mask register conflict with
  BF16 helpers, etc.

For the type pairs jit_kernel actually needs today — f32 ↔ f32 and
f32 ↔ u8 — the load/store paths are ~50 lines of focused code per pair.
Expressed as DSL-native primitives:

- `load(variable<f32[N]>&, variable<u8*>, length)` — f32←u8 widening
- `load(variable<f32[N]>&, variable<f32*>, length)` — plain masked/full
- `store(variable<u8*>, variable<f32[N]>, length, saturate_mode)` — f32→u8
- `store(variable<f32*>, variable<f32[N]>, length)` — plain masked/full

Total: ~250 lines of DSL-owned code, covering the color_convert-scale
scope. No dependency on `jit_load_emitter`, no eager emission inside
jit_kernel's IR mode, no pipeline-composition friction.

**What's given up**: BF16, F16, int8-quantization type pairs. Kernels that
need those types drop to the **escape hatch**: eager mode with direct
calls to the legacy emitters, interleaved with DSL code. The IR-mode
optimization doesn't cover those sections, and the kernel author takes
responsibility for register allocation across the boundary. This is the
same pattern as "drop to raw `uni_v*` for instructions the DSL doesn't
wrap" — always available, never the default.

**Principled split**:

- **DSL-native load/store** for the type pairs jit_kernel owns (today:
  f32, u8; extend as kernel demand materializes). These compose cleanly
  with IR mode. Correct on all target ISAs. ~50-100 lines per type pair.
- **Legacy emitters as an escape hatch** for exotic type pairs (BF16, F16,
  future f8/f4). Accessed from eager code, composes with DSL code via the
  standard "leave IR mode, do raw thing, return to IR mode" pattern.
- **No hard dependency** from jit_kernel on the legacy emitters for its
  common path. They are a library we compose with, not a foundation we
  build on.

This rule also defangs a sharp edge called out in the "orthogonal perf
gap" subsection (AVX2 masked-store corner case, scalar-tail fallback for
partial stores on AVX2 vs. maskstore on AVX-512). That concern lives
inside the DSL-native load/store primitives, where it's under our control
and can be tested directly, rather than being bundled into a centralized
emitter that many other kernels also use.

### IR mode vs. eager mode — explicit modes

Even after the load/store split, not every kernel benefits from IR mode.
Kernels with simple register footprints — loop-heavy code, few live
values, straightforward dataflow — gain nothing from IR and would only
pay the abstraction cost (debugging via IR dump instead of xbyak). The
sensible shape is **two explicit modes**, picked per-kernel:

- **Eager mode** (today's default). Every DSL call emits immediately.
  Fast to understand, fast to debug. Composes with legacy emitters
  without friction. Used by the majority of kernels.
- **IR mode** (opt-in, narrow). DSL calls record into a per-kernel IR.
  Pass 2 allocates and emits. Used by kernels fighting register pressure,
  typically in the self-contained math-heavy regions; load/store uses
  DSL-native primitives; exotic type conversion drops to eager.

The kernel author picks per-kernel (or per-region). There is no "upgrade
all kernels to IR mode" roadmap, because eager is fine for most work.
The IR mode is the hammer you reach for when a specific kernel shows
pressure symptoms in the baseline (cheap-tier stats).

### What we've shipped so far

- **Move-constructor** on `variable_base`: already copies the `shared_reg`
  cheaply (shared_ptr copy is refcount bump, no emission).
- **`operator=(variable&&) const noexcept`** on `variable<T[N],
  register_tag>` (shipped in the by-value refactor): steals the rhs's
  `shared_reg` instead of emitting `vmovups`. Covers the LHS-side elision
  for every `r = expr_returning_variable` pattern. `_reg` marked `mutable`
  so const methods can rebind without breaking the "const handle" model.
- **Symmetric copy/move-assign on scalar `variable<T, register_tag>`**
  (shipped in the color_convert refactor): the scalar specialization was
  missing `operator=(const variable&)` and `operator=(variable&&)` that the
  vector case already had. `store_interleaved3` needed `p = dst;` to work;
  without these overloads the compiler fell through to the implicitly
  deleted copy-assign and failed. The two new operators mirror the vector
  form — copy-assign emits one `mov`, move-assign rebinds the handle with
  zero emitted code. Closes a symmetry gap that would've hit every future
  scalar-pointer cursor pattern the same way.
- **`store_interleaved3(dst, a, b, c[, count])`** (shipped in the
  color_convert refactor): public DSL primitive for 3-way interleaved stores.
  Full-width overload lowers on x86 to the private `interleave_regs` helper
  (3×vpermps + 6×vblendps) followed by three plain stores; the tail overload
  stack-spills the full interleave and then `copy<T>(dst, stack, count*3)`
  to emit a variable-length partial store. API shape matches Highway's
  `StoreInterleaved3` and `std::datapar::simd_unchecked_store` — caller
  advances `dst` externally, the function is pure memory-side-effect with
  `void` return. `interleave_regs` stays a private helper because reg-only
  3-way interleave isn't exposed by any modern portable SIMD library
  (Highway has no reg-only form; std::datapar P0918 is 2-way only) —
  the fused store is the shape that maps to ARM `VST3` / SVE `ST3` / RVV
  `vsseg3`, and the reg-only form is strictly worse on those ISAs.

What's still missing:

- **Liveness-accurate allocation.** The shared_reg pool releases a slot
  only when the C++ `variable` goes out of scope, not at the last actual
  use inside the emitted sequence. The color_convert investigation
  quantified this cost directly — see "Empirical evidence" below.
- **Seed-move elision inside value-producing ops.** `fma(a, b, c)` still
  emits `vmovups res, c; vfmadd231ps res, a, b`. The seed move is
  move-eliminated by Haswell+ register renaming (zero ALU uops at runtime)
  but still costs a decoded uop and counts as an "allocation" for our
  pressure tracking. Move-hint overloads would close this for callers
  willing to annotate.
- **Liveness-aware fresh allocation.** When `var<float[N]>()` picks a
  register from the free pool, it has no preference for a just-freed slot
  vs. a cold one. A smarter pool could prefer reusing the most recently
  freed register to improve locality. ~20 lines. Minor win.
- **Pool cap at 16 regs on AVX-512.** `jit_kernel.cpp:334` populates
  `_free_rmmregs` from the GPR index range `RAX..R15` (16 entries). The
  `xmmregs()`/`ymmregs()`/`zmmregs()` arrays are also sized at 16 and
  only include `xmm0..xmm15` / `ymm0..ymm15` / `zmm0..zmm15`. On AVX-512
  the CPU has `zmm0..zmm31` — **half the architectural register file is
  ignored by the DSL**. A 10-line fix (separate pool size from GPR count,
  extend the zmm array to 32) would land for every JIT kernel, not just
  the ones we're actively refactoring.

### Empirical evidence — color_convert refactor (2026-04-11)

The color_convert refactor exercised the pressure question end-to-end and
produced concrete numbers. Worth recording because it converted "the
pressure gap might matter" from a theoretical concern into a measured
blocker for aggressive hoisting.

**The three-pass dump analysis** on a 512-bit AVX-512 build showed the
`yuv_to_rgb` inner loop re-emits the same loop-invariant setup every
pixel:

- **~21 instructions per iteration** wasted on coefficient broadcasts
  (`mov r8, r15; add r8, offset; vbroadcastss zmm, [r8]` × 8 coefficients).
  These constants are BT.601 coefficients and the `[0, 255]` clamp bound —
  they never change across iterations.
- **~15 instructions per iteration** wasted on permute-index-table reloads
  (`movabs r8, <index_table_addr>; vmovdqu32 zmm, [r8]` × 3). The tables
  are compile-time constants and identical every iteration.
- **~12 instructions per iteration** wasted on k-mask construction
  (`mov r8d, imm; kmovw k1, r8d` × 6). The immediates are fixed.

Total: ~48 instructions per pixel that any real register allocator with
LICM would hoist outside the loop automatically. The current DSL doesn't —
it emits what you write, and the programmer wrote the broadcasts inline.

**Attempting manual hoisting at the source level triggered register
pressure failures**:

1. First attempt — hoist 9 coefficients as a `yuv_coeffs<N>` struct
   above the foreach loop. Peak rose to **~19 concurrent live** inside
   `interleave_regs`'s blend chain (`ap, bp, cp, out0, out1, t_inner, out2`
   = 7 working + 9 hoisted + 3 original r/g/b kept alive as const-ref
   params through store_interleaved3). Exceeded the 16-cap pool,
   `reserveReg` threw, Xbyak bailed with a misleading "label is not set
   by L()" during cleanup.

2. Second attempt — same struct but built *inside* `yuv_to_rgb` (released
   before `store_interleaved3` runs). Working peak inside `yuv_to_rgb`
   alone reached **~16 briefly during the clamp stage**, exactly at the
   pool limit. Some additional transient pushed over, same failure.

3. Working result — by-value mutation pattern without hoisting. Working
   peak **~6 inside `yuv_to_rgb`** (3 for y'/u'/v' rebound in place +
   3 for r/g/b). Passes all 18 ConvertColor smoke tests. Emits 2368 bytes
   of kernel code per converter, same as baseline.

**What the measurement teaches**:

- The theoretical peak for the full pipeline (load → math → FMA → clamp →
  interleave → store, with all YUV coefficients hoisted) is **19** under
  naive accounting, dropping to **~14-16** if the allocator reclaims dead
  values at their last use instead of at scope exit. The DSL's
  refcount-scoped allocator is the *only thing* blocking aggressive
  hoisting.
- **~6 vector registers** of the current working peak are dead-held — they
  hold values whose last use has already emitted, but whose C++ handles
  are still in scope. The old fluent/in-place style dodged this by
  rebinding the same variable name through the chain, but it made
  functions impure and broke composition. The by-value pattern is
  strictly better for API shape, but exposes the liveness gap.
- The **16-register pool cap** is an artifact of the DSL init loop, not
  the architecture. On AVX-512 we're leaving 16 physical registers on the
  table. Bumping the pool would relax pressure across every JIT kernel,
  not just the ones we're touching.

Conclusion: the lightweight register allocator is the right investment.
Not because the cost accounting changed, but because the empirical evidence
from one realistic kernel showed that (a) the pressure gap is real and
reproducible, (b) manual hoisting is fragile and doesn't compose with the
by-value style we want, and (c) the DSL pool is artificially tight on
AVX-512 anyway. The cheap-tier debug counters would have told us the
pressure was close to the limit, but the fundamental fix — making dead
variables *actually dead* in the allocator — requires liveness analysis,
which requires the two-pass IR mode.

### Recommended ordering

The color_convert investigation collapsed several earlier "gate on
measurement" steps — we have the measurement now, and the pressure gap is
real. Revised order, committing to the regalloc path:

1. **Pool cap fix**. Separate the vector-register pool size from the GPR
   loop limit in `jit_kernel.cpp:334`. On AVX-512 populate `_free_rmmregs`
   with 32 entries and extend the `zmmregs()` array to `zmm0..zmm31`. Keep
   16 on AVX2 / SSE (those are the architectural limits). ~10-20 lines,
   trivial risk, benefits every JIT kernel. Do this first — it's the only
   piece of the roadmap that can ship without any new infrastructure and
   unblocks some pressure on its own.
2. **Cheap tier debug counters** — peak tracker + allocation counter
   guarded under `ENABLE_DEBUG_CAPS`. ~50 lines. Still worth doing even
   though we're committing to the regalloc path: gives us regression
   detection when kernels drift close to the limit, and validates the
   regalloc's output once it lands.
3. **DSL-native load/store primitives for f32 and u8.** ~250 lines. Lands
   independently of the regalloc work and eliminates the `jit_load_emitter`
   composition friction that would otherwise complicate IR mode's
   entry/exit boundaries. Worth doing regardless of regalloc.
4. **IR mode — lightweight register allocator (committed)**. ~900 lines,
   roughly one week. Two-pass: record DSL calls into an IR, run
   linear-scan allocation over the recorded intervals, emit xbyak with
   the allocated physical registers. Opt-in per kernel via a mode flag;
   eager mode stays the default. Explicit non-goals: no CSE, no LICM, no
   constant folding, no instruction selection beyond what DSL primitives
   already committed. **One DSL call = one emitted instruction**, just
   with liveness-correct physical register assignment.

   Scope breakdown:
   - Linear-scan allocator (Poletto & Sarkar 1999): ~150 lines.
   - IR data structures (op list, abstract value ids): ~200 lines.
   - IR → xbyak lowering (mechanical substitution): ~200 lines.
   - Control-flow handling (`_if`/`_then`/`_else`/`foreach`): ~200 lines.
     This is the non-trivial part — interval splitting across branches,
     spill/reload decisions at merge points.
   - Debug dumping for the IR: ~100 lines.

   Ship with IR mode enabled on `color_convert` (the kernel that
   motivated it) and validate end-to-end: the dump should show the
   per-iteration broadcasts and permute loads hoisted to the pipeline
   preamble, clamp bounds reused across channels, and peak live count
   matching the theoretical floor.
5. **Move-hint overloads**. Only if a kernel still shows pressure after
   the regalloc lands. Previously recommended before IR mode as a
   stepping stone; now the regalloc subsumes most of what move-hints
   would've bought us. Keep in the back pocket for cases where the
   author wants an explicit release annotation (e.g. across function
   boundaries the IR can't see).
6. **Expression templates**. Stays explicitly off the roadmap. The
   regalloc gives us liveness-accurate allocation and dead-value release
   at last use; expression templates would add single-statement fusion
   on top. Revisit only if a specific measurement shows IR mode is
   insufficient.

Phase 1 / 2 / 3 of the main DSL roadmap do not depend on the regalloc
track and can ship independently. But the regalloc track itself is now
the highest-leverage improvement to kernel code quality — the
color_convert empirical evidence shows ~48 instructions per pixel of
wasted loop-invariant setup that the regalloc would eliminate directly,
and it unblocks the by-value-everywhere idiom we want in the DSL.

### What IR mode is explicitly NOT

Worth stating up-front to prevent scope creep:

- **Not a compiler**. No CSE, no constant folding, no LICM, no dead-code
  elimination, no pattern matching, no instruction selection beyond what
  the DSL primitives already committed to.
- **Not a default**. Kernels stay on eager mode unless measurement shows
  they benefit from IR mode.
- **Not a path to std::datapar parity**. std::datapar users benefit from
  the full C++ compiler middle-end (inlining, autovectorization refinement,
  scalar CSE, etc.). IR mode here only matches **register allocation
  quality**, not the full optimizer stack. Real parity with std::datapar
  code quality requires MLIR or LLVM integration, which is the strategic
  decision explicitly declined in the "Why not MLIR" section.
- **Not a replacement for `jit_load_emitter` and friends**. The legacy
  emitters continue to handle BF16, F16, F8/F4, and other exotic type
  pairs. IR mode accesses them via the escape hatch (leave IR mode, do the
  thing eagerly, return to IR mode). The kernel author takes responsibility
  for register allocation across the boundary.
- **Not something every kernel needs**. The whole premise is that register
  allocation is the one thing a kernel author cannot easily manage
  manually. Kernels that don't have a pressure problem don't need the
  tool. Measure first.

### IR mode — implementation decisions (emerged during color_convert port)

These are design decisions that weren't in the original plan but were forced
by implementation experience. They should be treated as load-bearing
constraints, not accidental choices.

**GPRs are NOT IR-managed.** Only vector registers go through the IR
allocator. GPRs are allocated eagerly via the existing `var<size_t>()`
mechanism and must stay alive through lowering. Emit closures must not
call `var<>()` or `reserve<>()` — the GPR pool state at lowering time
differs from recording time. This means pointer arithmetic, loop counters,
and constant-table addresses are all outside the IR's jurisdiction.

**Two-phase recording/lowering model.** DSL calls during `begin_ir()` /
`end_ir()` record `Op` structs into an op list. `end_ir()` runs
`compute_intervals` → `linear_scan` → lowering (walking the tree, calling
emit closures with physical register assignments). Any state mutation
during lowering (reserve, free, push, pop) is invisible to the IR
allocator. Emit closures should ideally be pure functions of their
`EmitContext`.

**Nested regions for control flow.** `ir_if` and `foreach` create nested
`IR` bodies via `_ir->region()` / `_ir->loop()`. The cursor
save/restore in `region()` is critical — without it, nested control flow
(ir_if inside foreach) clobbers the outer cursor. Loop regions extend
intervals of pre-loop values; branch regions do not.

**`Xbyak::Address` captures register indices, not values.** If an
`Address` is constructed from a temporary GPR variable (via
`variable::operator+`), the Address survives but the GPR is freed. At
lowering time the index refers to a reused register. Always use stable
GPRs (member variables, `_consts.reg()`) or raw `Xbyak::RegExp` when
constructing addresses for emit closures.

**Rematerialization for loop-carried broadcasts.** When register pressure
exceeds the pool, the allocator evicts cheap ops (broadcasts with no
reads) and clones them before their next use. Clone intervals use actual
last-read (not the original's end) to keep pressure minimal. Wrap-around
search handles values whose next use is earlier in the loop body (next
iteration). Rewriting victim→clone must recurse into nested REGION bodies.

**`param1` (rdi) is the only safe scratch GPR for emit closures.** It's
excluded from the GPR pool by `isRegAllocable()`. All other GPRs (including
`rax`) can be allocated as loop counters or variables. `param1` holds the
params pointer, so it must be saved/restored if clobbered — but during IR
lowering, no emit closure accesses it (all param fields were loaded into
their own GPRs during `arg()` calls before `begin_ir()`). The post-IR tail
code uses `argPtr()` which needs `param1`, so the save/restore is necessary.

**Tail handling is eager, not IR.** After `end_ir()`, the remaining
`width % N` pixels are processed in eager mode via masked loads/stores.
This is correct and simple but means the IR loop only executes when
`width >= N`. On AVX-512 with N=16, small test widths (e.g. 10) produce
zero IR iterations — the entire conversion runs as tail.

## Sharp edges

Worth documenting because they will bite.

- **Tail handling.** `color_convert.cpp:472` has a separate tail path with a
  scalar `width != 0` check and partial load/store. Closure chains encode the
  body once; the tail wants to reuse that body with different load/store
  stages. Workable only if **stages take their load/store as parameters, they
  don't capture them.** Easy to get wrong; call it out in the stage‑writing
  guide.
- **Mid‑stage control flow.** `_if(color_format == 0)._then(...)` inside
  `yuv_to_rgb` (`color_convert.cpp:270`) is fine as code inside a stage
  lambda. Resist the temptation to lift the `if` to *stage selection at emit
  time* for runtime values — that just emits both branches, which `_if/_then`
  already does, only worse.
- **Premature stage reuse.** The moment `yuv_to_rgb` has one caller, resist
  turning it into a "library stage". Two callers, then library. Same reason
  most existing `jit_kernel` helpers have one caller each.
- **`variable` lifetime across lambda captures.** `variable_base`'s copy
  constructor (`jit_kernel.hpp:967`) copies the `shared_reg`, which keeps the
  register live via refcount. This is usually what you want, but a stage that
  captures a variable by value in a lambda will silently extend its lifetime.
  Document this, or make `variable` move‑only and require explicit
  `std::move` into returns.
- **Error messages.** With concepts, template errors are tolerable. Without
  concepts, they're Eigen‑class. **Use concepts.**
- **Tempting over‑abstraction.** Imperative xbyak with named variables and
  `foreach`/`_if` is already readable and steppable. Every feature added to
  the DSL should be justified by a concrete kernel that becomes clearly
  better. Delete aggressively when it doesn't.

## Proposed ordering

1. **Phase 1 — vector operators.** Shipped: `operator+ - *` on
   `variable<float[N]>`, plus an explicit `operator=(const variable&)` to
   unsuppress the copy-assignment that the move constructor implicitly
   deleted. `yuv_to_rgb` math block in color_convert rewritten from ~20
   `uni_v*` lines to ~9 lines of expressions. Acceptance test (is it
   clearly more readable?) is still the gating question before anything
   further.
2. **Load-and-construct factory overloads.** Shipped as a post-Phase-1
   follow-up: `k.var<float[N]>(src_ptr)`,
   `k.var<float[N]>(src_ptr, compile_time_length)`, and
   `k.var<float[N]>(src_ptr, runtime_length)`. Color_convert's four
   `var + load` pair sites (nv12 + i420 hot paths and tails) collapsed to
   single-expression form. Motivated by the simd wrapper alignment goal
   (see "API alignment" section).
3. **Next alignment items** — follow the "Match when a caller appears"
   table in the API alignment section. Each row is a small, scoped PR; do
   not batch them speculatively.
4. **`tile` + `Stage` + `|`.** ~100 lines of pure templates. Ship only
   after Phase 1 acceptance passes *and* a concrete second caller exists.
5. **Scratch pool helper.** Small, only with the Stage layer.
6. **Free-function `load_*`/`store_*` stages.** Grow organically per kernel.
7. **Port `color_convert` as the proving ground for Stages.** If the
   ported version isn't clearly better than the current file, **stop and
   delete the Stage layer.** Keep Phase 1 and the factory overloads.
8. **Phase 3 only if triggered.** Diff the three `jit_uni_eltwise_generic`
   files first. If they're structurally the same, proceed; otherwise don't.
9. **Orthogonal, perf-driven:** upgrade `jit_load_emitter` /
   `jit_store_emitter` tail paths to use hardware maskload/maskstore
   instead of stack-bounce. See the "Orthogonal perf gap" subsection of
   the API alignment section. Do this only when a hot kernel motivates
   it, not for color_convert. **Reconsidered under the load/store split**:
   if the DSL-native load/store track in the "Register allocation" section
   ships first, this item may dissolve — the DSL would own its own masked
   load/store implementations and this perf gap moves under our control
   rather than being a centralized emitter concern.
10. **Register-allocation track (orthogonal to Phases 1–3).** See the
    "Register allocation and pressure analysis" section for the full
    ordering within this track:
    - Cheap tier (peak tracker + allocation stats) — ship first, measure
      color_convert and neighboring kernels to get a baseline.
    - Move-hint overloads (~150 lines) — narrow, opt-in, closes the
      by-value pressure gap for kernels that opt into `std::move` at the
      call site.
    - DSL-native load/store for f32 + u8 (~250 lines) — unlocks IR mode
      and delivers independent value by removing dnnl emitter friction
      from color_convert's common path.
    - IR mode (~900 lines, one week) — narrow-scope register allocator,
      opt-in per kernel, explicitly not a compiler. Only ships after
      measurement shows a kernel benefits.
    Each sub-step is independently valuable; none is required for
    Phases 1–3. Escalate only when measurement demands it.

Phase 1 stands on its own merits regardless of whether Phase 2 or 3 ever
happen. If Phases 2–3 turn out to be over‑engineered in practice, Phase 1
still pays for itself every time anyone writes vector math in this framework.

## Why not MLIR (for this scope)

MLIR's `vector` + `arith` + `scf` + `x86vector` / `arm_neon` / `arm_sve`
dialects can express everything this DSL can, plus scalable vectors,
predication, autotuning, and cross‑arch lowering. ExecutionEngine + an
on‑disk object cache gives a JIT story comparable to what `jit_kernel`
produces (function pointer at the call site, heavy work amortized at
`compile_model` time).

It is still the wrong tool for this problem:

- **It's a compiler stack, not a library.** Adopting MLIR means building
  LLVM+MLIR, a pass pipeline, a lowering strategy, and a cache invalidation
  scheme keyed on (source hash, pipeline version, LLVM version, CPU features,
  flags). That's a strategic dependency, not a refactor.
- **It inverts the control model.** The whole premise of this DSL is that
  the developer controls fusion, register pressure, and scheduling. MLIR's
  value proposition is the opposite — you write the math, the pass pipeline
  decides. The moment you want to hand‑pick a specific `vfmadd` variant or
  emit exactly one `vpshufb`, you're fighting MLIR, not using it.
- **It's a cross‑project decision.** If CPU ukernels move to MLIR in
  OpenVINO, that call belongs in the plugin strategy conversation, not
  inside a refactor of one header. Don't be the second uncoordinated MLIR
  adoption in the same codebase.
- **Debuggability regresses.** Wrong‑output bugs in hand xbyak are
  disassemble‑and‑grep; wrong‑output bugs in an MLIR lowering pipeline are
  `-mlir-print-ir-after-all` + pass bisection. Different tooling, different
  skill set.

The natural end state if MLIR adoption does happen elsewhere in OpenVINO is
**not** "rewrite `jit_kernel` as a dialect". It's "let the graph compiler
lower straight to `vector` dialect for the 90% of kernels it can handle, and
keep xbyak as the escape hatch for the 10% that need cycle‑level control".
Phases 1–3 above are consistent with that end state; they make the escape
hatch nicer without committing to the compiler stack.
