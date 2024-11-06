// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

// Model optimization patterns. Triggered by the plugin at the very top
namespace patterns {
namespace opt {

class DQMatMulCWi : public ov::pass::MatcherPass {
public:
    DQMatMulCWi();
};

struct Context {
    std::string pmm_dims;
    bool is_spatial = false;

    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    using NPtr = std::shared_ptr<ov::Node>;

    using Axes = std::vector<std::size_t>;
    std::map<PPtr, Axes> closures_to_permute;
    void permute(PPtr orig_param, const Axes& order);

    std::set<PPtr> closures_to_f16;
    void to_f16(PPtr orig_param);

    using O = ov::Output<ov::Node>;
    struct DQParMM {
        PPtr w, s;
        NPtr mm;
    };
    using DQParMMs = std::vector<DQParMM>;
    std::map<std::pair<O, std::size_t>, DQParMMs> par_dq_mms;
    void register_parallel_matmul(O multiply, std::size_t axis, DQParMM&& mm);

    std::map<PPtr, std::pair<ov::ParameterVector, std::size_t>> params_to_concat;
    PPtr concat(ov::ParameterVector&& v, std::size_t dim);

    struct DQUnpack {
        PPtr w, z, s;
    };
    std::map<PPtr, DQUnpack> params_to_unpack;
    PPtr unpack(PPtr w, PPtr z, PPtr s, ov::element::Type type);
    PPtr unpack(PPtr w, PPtr s, ov::element::Type type);

    struct Gather {
        PPtr pnew, pold, pids;
    };
    std::optional<Gather> params_to_gather;
    PPtr host_gather(PPtr w, PPtr ids);

    using Ref = std::reference_wrapper<Context>;
};

class DQMatMulGQi : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQi(Context::Ref ctx);
};

class DQMatMulGQ2i : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQ2i(Context::Ref ctx);
};

class DQMatMulGQiP : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQiP(Context::Ref ctx);
};

class DQMatMulGQ2iP : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQ2iP(Context::Ref ctx);
};

class DQParMMGQ : public ov::pass::MatcherPass {
public:
    explicit DQParMMGQ(Context::Ref ctx);
};

void mergeParallelMatMuls(const std::shared_ptr<ov::Model>& m, Context& ctx);

// Gather-related passes

class DQLiftGatherAsymCW : public ov::pass::MatcherPass {
public:
    DQLiftGatherAsymCW();
};

class DQLiftGatherSymCW : public ov::pass::MatcherPass {
public:
    DQLiftGatherSymCW();
};

class DQLiftGatherSymGQ : public ov::pass::MatcherPass {
public:
    DQLiftGatherSymGQ();
};

// Head vocab unpacks

class DQUnpackDictGatheru : public ov::pass::MatcherPass {
public:
    DQUnpackDictGatheru(Context::Ref ctx);
};

class DQUnpackDictGatherGQi : public ov::pass::MatcherPass {
public:
    DQUnpackDictGatherGQi(Context::Ref ctx);
};

class HostGather : public ov::pass::MatcherPass {
public:
    HostGather(Context::Ref ctx);
};

class HostGatherDQ : public ov::pass::MatcherPass {
public:
    HostGatherDQ(Context::Ref ctx);
};

// Tail vocab unpacks

class DQUnpackDictMatMulCWu : public ov::pass::MatcherPass {
public:
    DQUnpackDictMatMulCWu(Context::Ref ctx);
};

class DQUnpackDictMatMulGQi : public ov::pass::MatcherPass {
public:
    DQUnpackDictMatMulGQi(Context::Ref ctx);
};

class CompressDictMatMulf32 : public ov::pass::MatcherPass {
public:
    CompressDictMatMulf32(Context::Ref ctx);
};

// Slice last Matmul
class SliceLastMatmul : public ov::pass::MatcherPass {
public:
    SliceLastMatmul();
};

class SliceLastMatmulAdd : public ov::pass::MatcherPass {
public:
    SliceLastMatmulAdd();
};

class SliceLastMatmulTranspose : public ov::pass::MatcherPass {
public:
    SliceLastMatmulTranspose();
};

class SliceLastMatmulMultiply : public ov::pass::MatcherPass {
public:
    SliceLastMatmulMultiply();
};

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
