# 🧠 AI ASSISTANT PRIMER: BALANCED TERNARY LOGIC

## A Comprehensive Reference for Code Assistants

**Version:** 1.0  
**Last Updated:** December 4, 2025  
**Scope:** Everything an AI needs to understand, implement, and reason about ternary logic

---

# 📑 TABLE OF CONTENTS

## QUICK NAVIGATION

| Section | Topic | Jump To |
|---------|-------|--------|
| **PART I** | [Foundational Concepts](#part-i-foundational-concepts) | Start Here |
| **PART II** | [Mathematical Framework](#part-ii-mathematical-framework) | Theory |
| **PART III** | [Data Structures & Encoding](#part-iii-data-structures--encoding) | Implementation |
| **PART IV** | [Arithmetic Operations](#part-iv-arithmetic-operations) | Algorithms |
| **PART V** | [NPN Classification](#part-v-npn-classification--canonicalization) | Classification |
| **PART VI** | [T-ISA Architecture](#part-vi-t-isa-instruction-set-architecture) | ISA Design |
| **PART VII** | [Code Examples](#part-vii-complete-code-examples) | Copy-Paste Ready |
| **PART VIII** | [Quick Reference Tables](#part-viii-quick-reference-tables) | Lookup |
| **APPENDIX** | [Glossary & Resources](#appendix-a-glossary) | Definitions |

---

# PART I: FOUNDATIONAL CONCEPTS

## 1.1 What is Balanced Ternary Logic?

### Core Definition

**Balanced Ternary Logic (BTL)** is a three-valued logic system where each digit (called a **trit**) can have exactly three values:

```
Trit Values:
  -1  (negative, often written as 'T' or '-')
   0  (zero, neutral)
  +1  (positive, often written as '1' or '+')
```

### Why "Balanced"?

The values are **symmetric around zero**: {-1, 0, +1}

This is different from **unbalanced ternary**: {0, 1, 2}

**Key Advantage:** Balanced ternary has natural sign representation—negative numbers don't need special encoding (no two's complement needed)!

### Information Content

```python
# A single trit carries:
import math
bits_per_trit = math.log2(3)  # ≈ 1.58496 bits

# Comparison:
# - 1 bit  = 2 states
# - 1 trit = 3 states (58% more information!)
```

---

## 1.2 Radix Economy: Why Ternary is Optimal

### The Radix Economy Theorem

For representing numbers, the **cost** of using radix r is:

```
Cost(r) = r × logᵣ(N) = r × ln(N) / ln(r)
```

**Minimum occurs at r = e ≈ 2.718**

Since radix must be an integer:
- Binary (r=2): Cost ≈ 2.885 × ln(N)
- **Ternary (r=3): Cost ≈ 2.731 × ln(N)** ← Winner!
- Quaternary (r=4): Cost ≈ 2.885 × ln(N)

**Conclusion:** Ternary is 5.6% more efficient than binary for number representation.

---

## 1.3 The Function Space

### Arity-2 Functions (Two Inputs)

```
Inputs: 3² = 9 combinations
Outputs per combination: 3 choices
Total functions: 3⁹ = 19,683
```

**Comparison with Binary:**
- Binary arity-2: 2⁴ = 16 functions
- Ternary arity-2: 19,683 functions
- **Ternary is 1,230× richer!**

### Arity-3 Functions (Three Inputs)

```
Inputs: 3³ = 27 combinations
Outputs per combination: 3 choices
Total functions: 3²⁷ = 7,625,597,484,987 (7.6 trillion!)
```

---

## 1.4 Key Terminology Quick Reference

| Term | Definition |
|------|------------|
| **Trit** | Ternary digit: {-1, 0, +1} |
| **Trit-word** | Sequence of trits (e.g., 80-trit word) |
| **NPN** | Negation-Permutation-Negation equivalence class |
| **Clone** | Set of functions closed under composition |
| **Functional Completeness** | Ability to express any function via composition |
| **T-ISA** | Ternary Instruction Set Architecture |
| **Canonical Form** | Lexicographically smallest NPN-equivalent |

---

# PART II: MATHEMATICAL FRAMEWORK

## 2.1 Ternary Truth Tables

### Basic Structure

An arity-2 ternary function is defined by a 9-entry truth table:

```
  x\y |  -1    0   +1
------|----------------
  -1  |  f₀   f₁   f₂
   0  |  f₃   f₄   f₅
  +1  |  f₆   f₇   f₈
```

Each fᵢ ∈ {-1, 0, +1}, giving 3⁹ = 19,683 possible functions.

### Function ID Encoding

Every function can be represented as a unique integer (0 to 19,682):

```python
def function_to_id(truth_table: list[int]) -> int:
    """Convert 9-trit truth table to function ID."""
    result = 0
    for trit in truth_table:
        result = result * 3 + (trit + 1)  # Map {-1,0,+1} → {0,1,2}
    return result

def id_to_function(func_id: int) -> list[int]:
    """Convert function ID to 9-trit truth table."""
    table = []
    for _ in range(9):
        table.append((func_id % 3) - 1)  # Map {0,1,2} → {-1,0,+1}
        func_id //= 3
    return table[::-1]
```

---

## 2.2 Fundamental Ternary Functions

### The "Big Five" Operations

```python
# 1. NEGATION (unary)
def tnot(a: int) -> int:
    """Ternary negation: -1↔+1, 0↔0"""
    return -a

# 2. MINIMUM (ternary AND)
def tmin(a: int, b: int) -> int:
    """Ternary AND: returns smaller value"""
    return min(a, b)

# 3. MAXIMUM (ternary OR)
def tmax(a: int, b: int) -> int:
    """Ternary OR: returns larger value"""
    return max(a, b)

# 4. CONSENSUS (majority vote for 3 inputs)
def tconsensus(a: int, b: int, c: int) -> int:
    """Returns value that appears at least twice, or 0 if all different."""
    s = a + b + c
    if s >= 2: return 1
    if s <= -2: return -1
    return 0

# 5. THREE-WAY COMPARISON
def tcmp(a: int, b: int) -> int:
    """Compare: returns -1 if a<b, 0 if a==b, +1 if a>b"""
    if a < b: return -1
    if a > b: return 1
    return 0
```

### Functional Completeness Theorem

**Theorem:** A set of ternary functions is **functionally complete** if it can express:
1. Negation (TNOT)
2. Minimum OR Maximum (TMIN or TMAX)
3. At least one non-linear function (e.g., TCONSENSUS)

**T-ISA is functionally complete** because it includes: TNOT, TMIN, TMAX, and TCONS.

---

## 2.3 NPN-Equivalence Theory

### What is NPN?

**NPN** = **N**egation-**P**ermutation-**N**egation

Two functions are NPN-equivalent if one can be transformed into the other by:
1. **N** (Input Negation): Negate any subset of inputs
2. **P** (Permutation): Reorder inputs
3. **N** (Output Negation): Negate the output

### Arity-2 NPN Group

```
Input Negations: 2² = 4 patterns (negate x, y, both, neither)
Input Permutations: 2! = 2 (swap x↔y or not)
Output Negations: 2 (negate output or not)

Total: 4 × 2 × 2 = 16 operations
```

**Result:** 19,683 functions collapse to **1,639 NPN-equivalence classes**

### Arity-3 NPN Group

```
Input Negations: 2³ = 8 patterns
Input Permutations: 3! = 6 permutations
Output Negations: 2 options

Total: 8 × 6 × 2 = 96 operations
```

**Result:** 7.6 trillion functions collapse to ~10⁵-10⁷ classes (exact count TBD by Project Atlas 7.6T)

---

## 2.4 Clone Lattice Structure

### Definition

A **clone** is a set of functions closed under:
1. **Composition**: If f,g ∈ clone, then f(g(...)) ∈ clone
2. **Projections**: All projection functions are in every clone

### The 18 Maximal Clones (Jablonski Classification)

For ternary logic, there are **18 maximal proper clones**:

| Clone | Description | Key Property |
|-------|-------------|-------------|
| M | Monotone | f(x) ≤ f(y) when x ≤ y |
| S | Self-dual | f(-x) = -f(x) |
| L | Linear | Modular structure |
| C₀ | Preserves 0 | f(0,0,...) = 0 |
| C₁ | Preserves +1 | f(1,1,...) = 1 |
| C₋₁ | Preserves -1 | f(-1,-1,...) = -1 |
| ... | (12 more) | ... |

**Application:** Knowing which clone a function belongs to enables optimization (e.g., monotone functions need no overflow checking).

---

# PART III: DATA STRUCTURES & ENCODING

## 3.1 Trit Representation in Binary

### Scheme A: Two-Bit Encoding (Simple)

```
Trit Value | Binary Code | Notes
-----------|-------------|------
    -1     |     00      | Lexicographically minimal
     0     |     01      | Central value
    +1     |     10      | Lexicographically maximal
    ??     |     11      | Reserved (NaN/undefined)
```

**Properties:**
- Simple to implement
- Negation: flip the high bit
- Zero detection: `code == 01`
- **Overhead:** 2 bits per trit (wasteful)

```python
class TritSchemeA:
    @staticmethod
    def encode(trit: int) -> int:
        return {-1: 0b00, 0: 0b01, 1: 0b10}.get(trit, 0b11)
    
    @staticmethod
    def decode(bits: int) -> int:
        return {0b00: -1, 0b01: 0, 0b10: 1}[bits]
```

### Scheme B: Dense Packing (Optimal)

Treat the trit sequence as a base-3 number:

```python
def pack_dense(trits: list[int]) -> int:
    """Pack trits into minimal bits. 80 trits → 127 bits."""
    result = 0
    for trit in trits:
        result = result * 3 + (trit + 1)
    return result

def unpack_dense(value: int, n_trits: int) -> list[int]:
    """Unpack bits to trits."""
    trits = []
    for _ in range(n_trits):
        trits.append((value % 3) - 1)
        value //= 3
    return trits[::-1]
```

**Bit Requirements:**
```python
import math
def bits_needed(n_trits: int) -> int:
    return math.ceil(n_trits * math.log2(3))

# Examples:
# 40 trits → 64 bits (fits in uint64)
# 80 trits → 127 bits (fits in uint128 with 1 bit overhead!)
```

---

## 3.2 The 80-Trit Word

### Why 80 Trits?

```
80 trits × log₂(3) = 126.9 bits ≈ 127 bits

Fits perfectly in a 128-bit register with 1.1 bits overhead!
```

### Word Configuration Table

| Trits | Exact Bits | Fits In | Overhead | Use Case |
|-------|------------|---------|----------|----------|
| 20 | 31.7 | 32-bit | 0.3 | Quarter-word |
| 40 | 63.4 | 64-bit | 0.6 | Half-word |
| **80** | **126.9** | **128-bit** | **1.1** | **Standard word** |
| 160 | 253.8 | 256-bit | 2.2 | Double-word |
| 320 | 507.6 | 512-bit | 4.4 | Quad-word (AVX-512) |

---

## 3.3 TernaryWord Class

```python
from typing import List

class TernaryWord:
    """80-trit balanced ternary word."""
    
    def __init__(self, trits: List[int]):
        if len(trits) != 80:
            raise ValueError("Must have exactly 80 trits")
        if not all(t in {-1, 0, 1} for t in trits):
            raise ValueError("Trits must be in {-1, 0, +1}")
        self.trits = trits
    
    @classmethod
    def from_int(cls, value: int) -> 'TernaryWord':
        """Convert integer to balanced ternary."""
        trits = []
        temp = abs(value)
        
        while temp > 0:
            remainder = temp % 3
            temp //= 3
            
            if remainder == 0:
                trits.append(0)
            elif remainder == 1:
                trits.append(1)
            else:  # remainder == 2 → carry
                trits.append(-1)
                temp += 1
        
        # Pad to 80 trits
        while len(trits) < 80:
            trits.append(0)
        
        # Handle negative
        if value < 0:
            trits = [-t for t in trits]
        
        return cls(trits[::-1])
    
    def to_int(self) -> int:
        """Convert to integer."""
        result = 0
        for trit in self.trits:
            result = result * 3 + trit
        return result
    
    def __repr__(self):
        symbols = {-1: '-', 0: '0', 1: '+'}
        s = ''.join(symbols[t] for t in self.trits[:20])
        return f"TernaryWord({s}... = {self.to_int()})"
```

---

# PART IV: ARITHMETIC OPERATIONS

## 4.1 Single-Trit Addition

### Truth Table

```
  a    b  | sum  carry
----------|------------
 -1   -1  |  +1    -1    ((-1)+(-1)=-2 → +1 with borrow)
 -1    0  |  -1     0
 -1   +1  |   0     0
  0   -1  |  -1     0
  0    0  |   0     0
  0   +1  |  +1     0
 +1   -1  |   0     0
 +1    0  |  +1     0
 +1   +1  |  -1    +1    ((+1)+(+1)=+2 → -1 with carry)
```

### Implementation

```python
def trit_add(a: int, b: int, carry_in: int = 0) -> tuple[int, int]:
    """Add two trits with carry. Returns (sum, carry_out)."""
    total = a + b + carry_in
    
    if total <= -2:
        return (total + 3, -1)  # Borrow
    elif total >= 2:
        return (total - 3, +1)  # Carry
    else:
        return (total, 0)      # No carry
```

---

## 4.2 Multi-Trit Addition (Ripple Carry)

```python
def ternary_add(a: TernaryWord, b: TernaryWord) -> TernaryWord:
    """Add two 80-trit words."""
    result = []
    carry = 0
    
    # Process from LSB to MSB
    for i in range(79, -1, -1):
        s, carry = trit_add(a.trits[i], b.trits[i], carry)
        result.append(s)
    
    return TernaryWord(result[::-1])
```

**Complexity:** O(n) where n = number of trits

---

## 4.3 Negation and Subtraction

```python
def ternary_negate(a: TernaryWord) -> TernaryWord:
    """Negate a ternary word. Simply flip all trits!"""
    return TernaryWord([-t for t in a.trits])

def ternary_subtract(a: TernaryWord, b: TernaryWord) -> TernaryWord:
    """Subtract: a - b = a + (-b)"""
    return ternary_add(a, ternary_negate(b))
```

**Key Insight:** No two's complement needed! Negation is trivial in balanced ternary.

---

## 4.4 Multiplication

### Single-Trit Multiplication

```python
def trit_multiply(a: int, b: int) -> int:
    """Multiply two trits. Result is always a single trit!"""
    # Truth table:
    # -1 × -1 = +1
    # -1 ×  0 =  0
    # -1 × +1 = -1
    #  0 × any = 0
    # +1 × -1 = -1
    # +1 ×  0 =  0
    # +1 × +1 = +1
    if a == 0 or b == 0:
        return 0
    return a * b
```

### Multi-Trit Multiplication

```python
def ternary_multiply(a: TernaryWord, b: TernaryWord) -> TernaryWord:
    """Multiply two 80-trit words (returns low 80 trits)."""
    # For simplicity, convert to int, multiply, convert back
    return TernaryWord.from_int(a.to_int() * b.to_int())
```

**Optimization:** Use Karatsuba for O(n^1.585) instead of O(n²)

---

## 4.5 Three-Way Comparison

```python
def ternary_compare(a: TernaryWord, b: TernaryWord) -> int:
    """Three-way compare. Returns -1, 0, or +1."""
    for i in range(80):
        if a.trits[i] < b.trits[i]:
            return -1
        if a.trits[i] > b.trits[i]:
            return +1
    return 0  # Equal
```

**Advantage:** Native three-way comparison (no separate equal/less/greater tests)!

---

# PART V: NPN CLASSIFICATION & CANONICALIZATION

## 5.1 The NPN Group Operations

### Arity-3 Example (96 Operations)

```python
from itertools import permutations

def apply_input_negation(func, negate_x: bool, negate_y: bool, negate_z: bool):
    """Create new function with negated inputs."""
    new_table = [0] * 27
    for idx in range(27):
        x, y, z = index_to_coords(idx)
        nx = -x if negate_x else x
        ny = -y if negate_y else y
        nz = -z if negate_z else z
        orig_idx = coords_to_index(nx, ny, nz)
        new_table[idx] = func.table[orig_idx]
    return TernaryFunction3(new_table)

def apply_output_negation(func):
    """Negate all outputs."""
    return TernaryFunction3([-t for t in func.table])

def apply_permutation(func, perm: tuple):
    """Permute input variables."""
    new_table = [0] * 27
    for idx in range(27):
        x, y, z = index_to_coords(idx)
        vars = [x, y, z]
        px, py, pz = vars[perm[0]], vars[perm[1]], vars[perm[2]]
        orig_idx = coords_to_index(px, py, pz)
        new_table[idx] = func.table[orig_idx]
    return TernaryFunction3(new_table)
```

---

## 5.2 Canonicalization Algorithm

```python
def find_canonical_form(func: TernaryFunction3) -> TernaryFunction3:
    """
    THE CORE ALGORITHM: Find lexicographically smallest NPN-equivalent.
    
    Time complexity: O(96 × 27) = O(2,592) per function
    """
    all_transforms = []
    all_perms = list(permutations([0, 1, 2]))  # 6 permutations
    
    # 8 input negation patterns × 6 permutations × 2 output negations = 96
    for perm in all_perms:
        for neg_x in [False, True]:
            for neg_y in [False, True]:
                for neg_z in [False, True]:
                    # Apply permutation, then input negation
                    f = apply_permutation(func, perm)
                    f = apply_input_negation(f, neg_x, neg_y, neg_z)
                    
                    # Without output negation
                    all_transforms.append(f)
                    
                    # With output negation
                    all_transforms.append(apply_output_negation(f))
    
    # Return lexicographically smallest
    return min(all_transforms, key=lambda f: f.to_int())
```

---

## 5.3 Function ID ↔ TernaryFunction3 Conversion

```python
def id_to_function(func_id: int) -> TernaryFunction3:
    """Convert ID (0 to 3²⁷-1) to function."""
    table = []
    for _ in range(27):
        table.append((func_id % 3) - 1)
        func_id //= 3
    return TernaryFunction3(table[::-1])

def function_to_id(func: TernaryFunction3) -> int:
    """Convert function to unique ID."""
    result = 0
    for trit in func.table:
        result = result * 3 + (trit + 1)
    return result
```

---

# PART VI: T-ISA INSTRUCTION SET ARCHITECTURE

## 6.1 Register Model

```
General Purpose Registers:
  T0  - T31  : 80 trits each (T0 hardwired to 0)

Special Registers:
  TPC    : Program counter (80 trits)
  TSP    : Stack pointer (80 trits)
  TFP    : Frame pointer (80 trits)
  TLR    : Link register (return address)
  TFLAGS : Status flags (27 trits)

Status Flags (TFLAGS):
  TZ : Zero     {-1: negative, 0: zero, +1: positive}
  TC : Carry    {-1: borrow, 0: none, +1: carry}
  TV : Overflow {-1: underflow, 0: none, +1: overflow}
  TS : Sign     {-1: negative, 0: zero, +1: positive}
```

---

## 6.2 Instruction Format

### Type-R: Register Operations

```
[Opcode: 8t][Rd: 5t][Rs1: 5t][Rs2: 5t][Func: 7t][Reserved: 10t]
  6561 ops    32 regs  32 regs  32 regs  2187 func

Total: 40 trits = 63.4 bits ≈ 64 bits
```

### Type-I: Immediate Operations

```
[Opcode: 8t][Rd: 5t][Rs1: 5t][Immediate: 22t]
  6561 ops    32 regs  32 regs  3²² values
```

---

## 6.3 Instruction Set Summary

### Arithmetic Instructions

| Mnemonic | Operation | Description |
|----------|-----------|-------------|
| `TADD Rd, Rs1, Rs2` | Rd = Rs1 + Rs2 | Ternary addition |
| `TSUB Rd, Rs1, Rs2` | Rd = Rs1 - Rs2 | Ternary subtraction |
| `TMUL Rd, Rs1, Rs2` | Rd = Rs1 × Rs2 | Multiplication (low 80t) |
| `TDIV Rd, Rs1, Rs2` | Rd = Rs1 ÷ Rs2 | Division (toward zero) |
| `TNEG Rd, Rs1` | Rd = -Rs1 | Negation |
| `TABS Rd, Rs1` | Rd = |Rs1| | Absolute value |

### Logic Instructions

| Mnemonic | Operation | Description |
|----------|-----------|-------------|
| `TAND Rd, Rs1, Rs2` | Rd = min(Rs1, Rs2) | Ternary AND |
| `TOR Rd, Rs1, Rs2` | Rd = max(Rs1, Rs2) | Ternary OR |
| `TNOT Rd, Rs1` | Rd = -Rs1 | Ternary NOT |
| `TMIN Rd, Rs1, Rs2` | Rd = min(Rs1, Rs2) | Minimum |
| `TMAX Rd, Rs1, Rs2` | Rd = max(Rs1, Rs2) | Maximum |
| `TCONS Rd, Rs1, Rs2, Rs3` | Rd = consensus | Majority vote |

### NPN Function Application

| Mnemonic | Operation | Description |
|----------|-----------|-------------|
| `TFUNC2 Rd, Rs1, Rs2, #npn` | Rd = f(Rs1, Rs2) | Apply any arity-2 function! |
| `TFUNC3 Rd, Rs1, Rs2, Rs3, #npn` | Rd = f(Rs1, Rs2, Rs3) | Apply any arity-3 function! |

**Power:** T-ISA can execute ANY of the 19,683 arity-2 functions directly!

### Comparison Instructions

| Mnemonic | Operation | Description |
|----------|-----------|-------------|
| `TCMP Rd, Rs1, Rs2` | Rd = sign(Rs1 - Rs2) | Three-way compare |
| `TSIGN Rd, Rs1` | Rd = sign(Rs1) | Extract sign |

### Control Flow

| Mnemonic | Operation | Description |
|----------|-----------|-------------|
| `TBEQ Rs1, offset` | if Rs1 == 0 | Branch if zero |
| `TBLT Rs1, offset` | if Rs1 < 0 | Branch if negative |
| `TBGT Rs1, offset` | if Rs1 > 0 | Branch if positive |
| `TBT Rs1, L-, L0, L+` | 3-way branch | Branch based on sign! |
| `TCALL offset` | TLR = PC; PC += offset | Function call |
| `TRET` | PC = TLR | Return |

---

## 6.4 Three-Way Branch: Unique to Ternary!

```assembly
# Binary style (inefficient):
  TCMP  T1, T2, T3      # Compare
  TBEQ  T1, label_eq    # Branch if equal
  TBLT  T1, label_lt    # Branch if less
  TJMP  label_gt        # Otherwise greater

# Ternary style (single instruction!):
  TCMP  T1, T2, T3
  TBT   T1, label_lt, label_eq, label_gt  # 3-way branch!
```

---

# PART VII: COMPLETE CODE EXAMPLES

## 7.1 Complete TernaryFunction3 Class

```python
"""
Complete arity-3 ternary function implementation.
Copy-paste ready for AI code assistants.
"""

from typing import List, Tuple
from itertools import permutations

class TernaryFunction3:
    """Arity-3 balanced ternary function (27-trit truth table)."""
    
    def __init__(self, table: List[int]):
        if len(table) != 27:
            raise ValueError(f"Need 27 entries, got {len(table)}")
        if not all(t in {-1, 0, 1} for t in table):
            raise ValueError("All values must be in {-1, 0, +1}")
        self.table = tuple(table)
    
    @staticmethod
    def coords_to_index(x: int, y: int, z: int) -> int:
        """(x,y,z) ∈ {-1,0,+1}³ → index 0-26"""
        return 9 * (x + 1) + 3 * (y + 1) + (z + 1)
    
    @staticmethod
    def index_to_coords(idx: int) -> Tuple[int, int, int]:
        """index 0-26 → (x,y,z) ∈ {-1,0,+1}³"""
        z = (idx % 3) - 1
        y = ((idx // 3) % 3) - 1
        x = ((idx // 9) % 3) - 1
        return x, y, z
    
    def evaluate(self, x: int, y: int, z: int) -> int:
        """Evaluate f(x, y, z)."""
        return self.table[self.coords_to_index(x, y, z)]
    
    def to_int(self) -> int:
        """Convert to unique integer ID (0 to 3²⁷-1)."""
        result = 0
        for trit in self.table:
            result = result * 3 + (trit + 1)
        return result
    
    def __eq__(self, other):
        return isinstance(other, TernaryFunction3) and self.table == other.table
    
    def __hash__(self):
        return hash(self.table)
    
    def __lt__(self, other):
        return self.to_int() < other.to_int()
    
    def __repr__(self):
        return f"TernaryFunction3({list(self.table[:9])}...)"
```

---

## 7.2 Complete Canonicalizer

```python
def apply_npn_transform(func: TernaryFunction3,
                        perm: Tuple[int, int, int],
                        neg_inputs: Tuple[bool, bool, bool],
                        neg_output: bool) -> TernaryFunction3:
    """Apply a single NPN transformation."""
    new_table = [0] * 27
    
    for idx in range(27):
        x, y, z = TernaryFunction3.index_to_coords(idx)
        vars = [x, y, z]
        
        # Apply permutation
        px, py, pz = vars[perm[0]], vars[perm[1]], vars[perm[2]]
        
        # Apply input negations
        if neg_inputs[0]: px = -px
        if neg_inputs[1]: py = -py
        if neg_inputs[2]: pz = -pz
        
        # Lookup original
        orig_idx = TernaryFunction3.coords_to_index(px, py, pz)
        value = func.table[orig_idx]
        
        # Apply output negation
        if neg_output:
            value = -value
        
        new_table[idx] = value
    
    return TernaryFunction3(new_table)


def find_canonical_form(func: TernaryFunction3) -> TernaryFunction3:
    """
    Find the canonical NPN-equivalent form.
    Returns the lexicographically smallest of all 96 transforms.
    """
    min_func = None
    min_val = float('inf')
    
    for perm in permutations([0, 1, 2]):  # 6
        for nx in [False, True]:          # 2
            for ny in [False, True]:      # 2
                for nz in [False, True]:  # 2
                    for no in [False, True]:  # 2
                        transformed = apply_npn_transform(
                            func, perm, (nx, ny, nz), no
                        )
                        val = transformed.to_int()
                        if val < min_val:
                            min_val = val
                            min_func = transformed
    
    return min_func
```

---

## 7.3 Complete Encoding Schemes

```python
import math

LOG2_3 = math.log2(3)  # ≈ 1.58496

class TernaryEncoding:
    """All three encoding schemes in one class."""
    
    # Scheme A: Two-bit symmetric
    @staticmethod
    def encode_a(trits: list) -> int:
        """Pack using 2 bits per trit."""
        result = 0
        mapping = {-1: 0b00, 0: 0b01, 1: 0b10}
        for trit in trits:
            result = (result << 2) | mapping[trit]
        return result
    
    @staticmethod
    def decode_a(value: int, n_trits: int) -> list:
        """Unpack 2-bit encoding."""
        mapping = {0b00: -1, 0b01: 0, 0b10: 1}
        trits = []
        for _ in range(n_trits):
            trits.append(mapping[value & 0b11])
            value >>= 2
        return trits[::-1]
    
    # Scheme B: Dense packing (optimal)
    @staticmethod
    def encode_b(trits: list) -> int:
        """Pack using log₂(3) bits per trit."""
        result = 0
        for trit in trits:
            result = result * 3 + (trit + 1)
        return result
    
    @staticmethod
    def decode_b(value: int, n_trits: int) -> list:
        """Unpack dense encoding."""
        trits = []
        for _ in range(n_trits):
            trits.append((value % 3) - 1)
            value //= 3
        return trits[::-1]
    
    @staticmethod
    def bits_required(n_trits: int) -> int:
        """Minimum bits needed for n trits."""
        return math.ceil(n_trits * LOG2_3)
```

---

# PART VIII: QUICK REFERENCE TABLES

## 8.1 Conversion Formulas

| From | To | Formula |
|------|----|---------|
| Trits to bits | `n_bits = ceil(n_trits × 1.58496)` | |
| Bits to trits | `n_trits = floor(n_bits / 1.58496)` | |
| Trit to 2-bit | `{-1: 00, 0: 01, +1: 10}` | |
| Index to (x,y,z) | `z=(i%3)-1, y=((i//3)%3)-1, x=((i//9)%3)-1` | |
| (x,y,z) to index | `9*(x+1) + 3*(y+1) + (z+1)` | |

---

## 8.2 Space Complexity

| Data Type | Size | Notes |
|-----------|------|-------|
| Single trit | 1.585 bits | Dense: 2 bits simple |
| 9-trit (arity-2 function) | 15 bits | 2-byte aligned |
| 27-trit (arity-3 function) | 43 bits | 6-byte aligned |
| 40-trit half-word | 64 bits | uint64_t |
| 80-trit word | 127 bits | uint128_t |
| 32×80-trit register file | 4096 bits | 512 bytes |

---

## 8.3 Time Complexity (Operations)

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Trit addition | O(1) | Lookup table |
| Word addition | O(n) | Ripple carry |
| Word multiplication | O(n²) | Naive |
| Word multiplication | O(n^1.585) | Karatsuba |
| NPN canonicalization | O(96 × 27) | Arity-3 |
| Function evaluation | O(1) | Table lookup |

---

## 8.4 Function Counts

| Arity | Inputs | Total Functions | NPN Classes |
|-------|--------|-----------------|-------------|
| 0 | 1 | 3 | 2 |
| 1 | 3 | 27 | 7 |
| 2 | 9 | 19,683 | 1,639 |
| 3 | 27 | 7.6 trillion | ~10⁵-10⁷ |
| 4 | 81 | 10³⁸ | Intractable |

---

## 8.5 NPN Group Sizes

| Arity | Input Negations | Permutations | Output Negations | Total |
|-------|-----------------|--------------|------------------|-------|
| 2 | 2² = 4 | 2! = 2 | 2 | 16 |
| 3 | 2³ = 8 | 3! = 6 | 2 | 96 |
| 4 | 2⁴ = 16 | 4! = 24 | 2 | 768 |
| n | 2ⁿ | n! | 2 | 2^(n+1) × n! |

---

# APPENDIX A: GLOSSARY

| Term | Definition |
|------|------------|
| **Balanced Ternary** | Three-valued system with {-1, 0, +1} |
| **BTL** | Balanced Ternary Logic |
| **Canonical Form** | Lexicographically smallest NPN-equivalent function |
| **Clone** | Function set closed under composition |
| **Consensus** | Majority vote function (returns value appearing ≥2 times) |
| **Functional Completeness** | Ability to express any function via composition |
| **Maximal Clone** | Largest proper clone (18 exist for ternary) |
| **NPN** | Negation-Permutation-Negation equivalence |
| **NPN Class** | Set of all NPN-equivalent functions |
| **Project Atlas 7.6T** | Effort to classify all 7.6T arity-3 functions |
| **Radix Economy** | Efficiency measure: r × logᵣ(N) |
| **T-ISA** | Ternary Instruction Set Architecture |
| **Trit** | Ternary digit: {-1, 0, +1} |
| **Truth Table** | Complete function definition as input→output mapping |

---

# APPENDIX B: FILE REFERENCE

## Primary Source Files

| File | Location | Purpose |
|------|----------|--------|
| VIRTUAL_TERNARY_PROCESSOR_FRAMEWORK.md | `/home/ubuntu/ternary_logic_wiki/public/virtual_ternary_processor/` | Complete theory + architecture |
| t_isa_spec_v1.md | Same directory | ISA specification |
| encoding_schemes.py | Same directory | Encoding implementations |
| ternary_arithmetic.py | Same directory | Arithmetic operations |
| btl_arity3_canonicalizer.py | `/home/ubuntu/` | NPN classification algorithm |
| PROJECT_ATLAS_7.6T_SUMMARY.md | `/home/ubuntu/` | Arity-3 classification project |

---

# APPENDIX C: COMMON PATTERNS

## Pattern 1: Iterate Over All Inputs

```python
# For arity-2 (9 combinations)
for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        result = function.evaluate(x, y)

# For arity-3 (27 combinations)
for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        for z in [-1, 0, 1]:
            result = function.evaluate(x, y, z)
```

## Pattern 2: Generate All Functions

```python
# Arity-2: 19,683 functions
for func_id in range(3**9):
    table = []
    temp = func_id
    for _ in range(9):
        table.append((temp % 3) - 1)
        temp //= 3
    function = TernaryFunction2(table[::-1])
```

## Pattern 3: Check Monotonicity

```python
def is_monotone(func):
    """Check if function is monotone (preserves order)."""
    for x1 in [-1, 0, 1]:
        for y1 in [-1, 0, 1]:
            for x2 in [-1, 0, 1]:
                for y2 in [-1, 0, 1]:
                    if x1 <= x2 and y1 <= y2:
                        if func.evaluate(x1, y1) > func.evaluate(x2, y2):
                            return False
    return True
```

## Pattern 4: Check Self-Duality

```python
def is_self_dual(func):
    """Check if f(-x,-y) = -f(x,y)."""
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            if func.evaluate(-x, -y) != -func.evaluate(x, y):
                return False
    return True
```

---

# APPENDIX D: PERFORMANCE BENCHMARKS

## Theoretical Comparison (Binary vs Ternary)

| Metric | Binary (80-bit) | Ternary (80-trit) | Ternary Advantage |
|--------|-----------------|-------------------|-------------------|
| Information | 80 bits | 126.9 bits | +58% |
| Addition gates | ~240 | ~1,120 | -367% (overhead) |
| Addition adjusted | 3 gates/bit | 8.8 gates/bit | -193% |
| Multiply | O(n²) | O(n²) | Same |
| Comparison | 2-way | 3-way native | +50% efficiency |

## Practical Guidelines

1. **Use ternary for:** Three-way decisions, signed arithmetic, ML inference
2. **Use binary for:** Bit manipulation, existing code interop
3. **Hybrid approach:** Ternary compute, binary I/O and storage

---

# APPENDIX E: COMMON MISTAKES

## Mistake 1: Confusing Balanced vs Unbalanced

```python
# WRONG: Unbalanced ternary {0, 1, 2}
trits = [0, 1, 2]  # This is NOT balanced ternary!

# CORRECT: Balanced ternary {-1, 0, +1}
trits = [-1, 0, +1]  # This IS balanced ternary
```

## Mistake 2: Forgetting Carry in Addition

```python
# WRONG: Just add and clamp
def bad_add(a, b):
    return max(-1, min(1, a + b))  # Loses information!

# CORRECT: Handle carry properly
def good_add(a, b, carry_in=0):
    total = a + b + carry_in
    if total <= -2:
        return (total + 3, -1)
    elif total >= 2:
        return (total - 3, +1)
    return (total, 0)
```

## Mistake 3: Wrong Index Formula

```python
# WRONG: Forgetting the +1 offset
def bad_index(x, y, z):
    return 9*x + 3*y + z  # Negative indices!

# CORRECT: Add 1 to map {-1,0,+1} to {0,1,2}
def good_index(x, y, z):
    return 9*(x+1) + 3*(y+1) + (z+1)  # Range [0, 26]
```

---

**END OF AI ASSISTANT PRIMER**

*This document synthesizes research from the Balanced Ternary Logic Working Group.*
*For questions or corrections, consult the source files listed in Appendix B.*
