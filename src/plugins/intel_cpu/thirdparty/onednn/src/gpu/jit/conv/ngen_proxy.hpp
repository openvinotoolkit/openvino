/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Proxy classes for selected nGEN functionality to minimize dependency on
// template-heavy nGEN headers, and reduce compilation time.

#ifndef GPU_JIT_CONV_NGEN_PROXY_HPP
#define GPU_JIT_CONV_NGEN_PROXY_HPP

#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace ngen_proxy {

enum class Access { Read, Write };

enum AddressModel {
    ModelInvalid,
    ModelBTS,
    ModelA64,
    ModelSLM,
};

enum AtomicOp { undef, fadd };

class Bundle {
public:
    Bundle() : bundle_id(any), bank_id(any) {}

    Bundle(int8_t bank_id_, int8_t bundle_id_)
        : bundle_id(bundle_id_), bank_id(bank_id_) {}

    bool operator==(const Bundle &other) const {
        return (bundle_id == other.bundle_id) && (bank_id == other.bank_id);
    }

    static const int8_t any = -1;

    int8_t bundle_id;
    int8_t bank_id;
};

class SBID {
public:
    SBID(int token = -1) : token(token) {}

    bool is_empty() const { return token == -1; }

    int token;
};

class InstructionModifier {
public:
    bool operator==(const InstructionModifier &other) const {
        return (is_atomic == other.is_atomic);
    }

    size_t get_hash() const { return ir_utils::get_hash(is_atomic); }

    InstructionModifier with_atomic() const {
        auto ret = *this;
        ret.is_atomic = true;
        return ret;
    }

    InstructionModifier with_sbid(const SBID &sbid) const {
        auto ret = *this;
        ret.sbid = sbid;
        return ret;
    }

    bool is_atomic = false;
    SBID sbid;
};

} // namespace ngen_proxy
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
