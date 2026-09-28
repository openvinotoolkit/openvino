# BRGEMM on the jit_kernel IR

Status as of 2026-09-29. A second BRGEMM code generator, written against
the `jit_kernel` IR, offered to oneDNN alongside its own three.

BRGEMM is the hardest test the IR has been put to: oneDNN's generator is
4096 lines, its register assignment is closed-form index arithmetic
(`accm(ld_block, bd, ld) = Vmm(max_effective_vregs - 1 - (bd*ld_block + ld))`),
and the blocking factors are *chosen so that the accumulator tile fits the
register file*. Register pressure there is an input to the shape of the
computation, not an outcome of allocation.

Sources:
- `src/plugins/intel_cpu/src/nodes/kernels/x64/brgemm_kernel_ir.{hpp,cpp}`
- `src/plugins/intel_cpu/tests/unit/brgemm_kernel_ir_test.cpp` — differential vs oneDNN
- `src/plugins/intel_cpu/tests/unit/brgemm_factory_test.cpp` — the hook's contract
- `src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/instances/x64/matmul.cpp` — `MatMulBrgemmBench`
- oneDNN submodule, branch `brgemm_jit_ir`, commit "cpu: x64: brgemm: allow an external kernel generator"

## How an alternative generator gets in

`brgemm_kernel_create()` dispatches to one of three built-in generators
chosen from the descriptor. A library layered on top cannot patch that
dispatch without oneDNN depending on it, and the dependency only runs one
way.

So oneDNN gained a registered factory it consults first — a function
pointer it holds, an implementation the caller owns. Three contract
decisions, each of which earned its keep later:

- **`status::unimplemented` means fall through**; any other non-success
  status propagates. A partial implementation declares its coverage by
  answering, not by omission, and a real bug in it cannot be mistaken for
  "oneDNN handled that one".
- **The factory returns a constructed but not yet created kernel**, so
  `create_kernel()` is called in one place for built-in and external
  alike and a generation failure is cleaned up identically.
- **The descriptor arrives finalized.** Blocking is already chosen by
  `brgemm_utils.cpp`; our generator consumes it. That is what makes a
  comparison meaningful — same shape, same blocking, two code
  generators — and it sidesteps the register-budget question entirely
  for now.

Two of the three x86 BRGEMM entry points did not need the hook at all:
`BrgemmKernel` (MHA/SDPA) and the snippets executor call
`brgemm_desc_init`/`brgemm_kernel_create` from plugin code. The hook is
what reaches kernels created *inside* oneDNN primitives, which is where
MatMul goes.

`brgemm_kernel_ir` implements `brgemm_kernel_t` and derives from
`jit_kernel`. `jit_generator_t::create_kernel()` and
`brgemm_kernel_t::create_kernel()` have the same signature, so one
override satisfies both bases, and `get_jit_generator()` returns `this` —
which means `ONEDNN_JIT_DUMP` and oneDNN verbose work on our kernel with
no new tooling.

## Three modes, and why the third exists

| `OV_JIT_IR_BRGEMM` | |
|---|---|
| unset / `0` | factory not installed |
| `1` | **offer** — take what is supported, oneDNN gets the rest |
| `2` | **force** — take what is supported, hard error on the rest |

`offer` is right for use and wrong for testing: a suite can pass green
while this generator never runs once, because every decline falls through
invisibly. Under `force` the same decline is an error naming the
descriptor and the reason.

**Force mode paid for itself on first use.** Pointed at MHA it reported
54 descriptors declined for `only f32 throughout` (they are s8) and 14
for `only brgemm_addr batching` (they are `brgemm_strd`, the snippets
path). MHA is lowered to a Subgraph, so *none* of its descriptors reach
this generator. Without force mode the MHA tests pass under `=1` and look
like coverage.

Pointed at MatMul instead, the census was 18 post-ops and 12 ld tails and
nothing else — a far deeper reach into the predicate. That redirected the
work from bf16 (the obvious guess) to the N tail.

## The supported slice

`unsupported_reason()` returns which check declined a descriptor, not a
bool, because the answer has to be reportable: it is the error text under
force and the skip text in the differential test.

Currently accepted: f32 throughout, AVX-512, `brgemm_addr`, alpha 1 and
beta 0, no post-ops/bias/scales, no AMX or depthwise, no quantization or
compensation or weight decompression, no virtual padding, no bd or rd
tail, `ld_block == 16`, and at most 8 column groups.

Each excluded line is a feature of the built-in kernel that has to be
reproduced and differentially tested before it can be deleted.

### Widening is a ratchet

The census reports only the *first* failing check, so the method is:
widen one check, re-run `OV_JIT_IR_BRGEMM=2`, read the new reason. Worth
running rather than predicting — the first prediction (dtype would lead)
was wrong.

MatMul's census, as the slice widened:

```
initial        18 post-ops · 12 ld (N) tail
after N tails  18 post-ops ·  4 rd (K) tail
```

## What the generator emits

Blocking comes from the descriptor: a `bd_block` x `ld_block2`
accumulator tile, `bdb` tiles down M, `rdb` steps of `rd_block` along K.

```
row_off = 0 ; c_cursor = C
foreach m_blk in bdb:                  runtime loop
  for each column group:               unrolled — they are not uniform
    acc[bd][ld] = 0
    foreach bs in BS:                  runtime loop
      A = batch[bs].ptr.A + row_off ; B = batch[bs].ptr.B
      foreach kb in rdb:               runtime loop
        for rd in rd_block:            unrolled
          load B columns
          for bd in bd_block:          unrolled — this is the register tile
            broadcast A
            acc[bd][ld] += a * b
        advance A, B
    store acc through c_cursor
  advance row_off, c_cursor
```

Rolled where the body is uniform, unrolled where it is not. The column
groups stay unrolled because the last one can be narrower and masked;
that is the rolled-plus-recorded-tail shape `foreach_with_epilogue`
already uses.

### N is covered by column groups plus a masked tail

`ldb` full 16-column blocks emitted `ld_block2` at a time, plus `ldb_tail`
leftover columns as one masked block, masking both the B load and the C
store. The store mask is the load-bearing one: without it the kernel
writes whole registers and scribbles past N.

**`ldb2` and `ldb2_tail` are unusable.** When `ld_block2` is shrunk to fit,
`brgemm_utils.cpp:219` leaves them at pre-shrink values — N=32 reports
`ldb2=0` while N=64 reports `ldb2=1` for the same `ldb == ld_block2`
relationship. Only `ldb`, `ldb_tail` and `ld_block2` can be trusted. Using
the obvious-looking fields would have produced a shape-dependent bug.

### One latent miscompile, caught before it shipped

An earlier version emitted a *single* group of `ld_block2` blocks. `ldb`
is the total and can exceed it: N=80 gives `ldb=5, ld_block2=4,
ldb_tail=0`, which passed every check and would have computed the first
64 columns and left the rest of C untouched. No differential shape reached
it. Found by reading how oneDNN's `ldb_loop` iterates, not by testing.

## What the IR needed

Four additions, all generally useful rather than BRGEMM-specific:

- **`IR::def_into` and `ir_accumulate`** — loop-carried accumulators. A
  running total could not be expressed at all: `vec_op` calls `def_tied`,
  which starts with `_next_value++`, and a loop body is recorded once, so
  `acc = acc + x` inside a loop reads the initial value every trip and
  discards its own result. It compiles, allocates, runs, and computes the
  wrong thing. See the journal.
- **`arch_emitter::gpr_load`** — a pointer-sized load, which the batch
  array needs.
- **`arch_emitter::gpr_add_reg`** — register plus register. A is re-fetched
  from the batch array every iteration, so its row offset cannot
  accumulate into the pointer and must be added afterwards; every other
  pointer operation took an immediate.
- **`Op::align`, `arch_emitter::align_to`, `preferred_loop_alignment`** —
  loop head alignment, see the journal.

## Register pressure — the question this was for

The 8x2 tile allocates **19 zmm**: 16 accumulators live across two nested
loops, 2 B columns, 1 broadcast. No spill, no rematerialization. oneDNN's
kernel for the same descriptor also uses 19. On the benchmark shape both
use 28 (24 accumulators, 3 B, 1 broadcast).

So the allocator reproduces by interference what oneDNN computes by hand,
at the pressure the hand arithmetic was designed for. That is the result
the whole exercise existed to get.

## Measurements

`MatMulBrgemmBench`, the two `IS_brgemm_smoke` shapes the generator
accepts, one thread, one stream. The benchmark passes under `=2`, so the
timings are of our kernel rather than a silent fallback — and the shapes
were chosen *because* they are accepted, since benchmarking a declined
descriptor compares oneDNN against itself and reports a dead heat.

Five interleaved repeats, `(7,32,120)x(3,7,120,50)`:

```
oneDNN   78 79 84 83 79     median 79
IR       82 83 84 84 82     median 83
```

About 5% behind. It was 8-10% before loop alignment.

### What moved the number, and what did not

| change | effect |
|---|---|
| loop alignment at 16 bytes | **89 -> 84 us**, half the original gap |
| loop alignment at 64 bytes (oneDNN's choice) | **96-98 us** — markedly worse |
| B prefetching, oneDNN's exact pattern | none |
| rolling the M loop | none (code 8617 -> 2240 bytes) |
| loop rotation | none |

Two of oneDNN's choices were actively wrong for this kernel when copied
directly. Both looked like obvious gaps in the instruction-mix diff.

### The inner loops are now equivalent

```
                 ours   oneDNN
vfmadd231ps        96       96
vbroadcastss       32       32
vmovups            12       12
prefetcht0          0       12
add                 3        2
branches            1        1
instructions      147      157
bytes             979     1068
```

Whole kernel: 2228 bytes against oneDNN's 2912.

The static profile matches or beats oneDNN's and the time does not
follow, so the residual is not anything countable in a listing. Next
instrument is `perf stat` on the two kernels — front-end stalls, port
pressure, memory — not more diffing.

## Open

1. **18 post-op descriptors.** `jit_uni_postops_injector` reserves and
   preserves registers by its own rules inside a region the allocator
   believes it owns. Needs the clobber-barrier op the IR does not have.
2. **4 rd (K) tail descriptors**, and the bd (M) tail that three
   differential shapes still skip for. The M tail is cheap now that the
   M loop is rolled — it is the epilogue slot of a loop that already
   exists.
3. **`brgemm_strd`**, the snippets path, which is where MHA's f32
   descriptors go.
4. **s8**, 54 MHA descriptors. Needs VNNI dot products and compensation
   passes; a different kind of work rather than a widening.
5. **AMX** — a fourth register class plus tile configuration held as
   machine state, the same problem as RVV's `vl`.
6. **The residual 5%**, unexplained.
