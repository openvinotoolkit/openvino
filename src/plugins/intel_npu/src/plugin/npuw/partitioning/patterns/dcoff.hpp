// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

struct Subgraph;  // Forward declaration
struct Function;  // Forward declaration

enum class DCOffMode : int {
    CAST_ONLY,
    CAST_SCALE,
};

namespace patterns {

// Common structures here

struct DCOFFParams {
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    std::unordered_map<PPtr, PPtr> scales;        // Closures: a scaling factor -> orig tensor
    std::unordered_map<PPtr, CPtr> zerops;        // Closures: orig tensor -> a zero point (yes, a reverse...)
    std::unordered_map<PPtr, PPtr> zerops_asymm;  // Closures: orig tensor -> an asymmetric zerop parameter
};

using DCOFFParamRef = std::reference_wrapper<DCOFFParams>;

struct ClosureRemap {
    std::vector<std::size_t> closure_remap;          // [new closure index] -> orig closure idx
    std::map<std::size_t, std::size_t> scale_remap;  // orig closure idx -> orig scale idx
    std::map<std::size_t, std::size_t> zerop_remap;  // orig closure idx -> orig asymm zero point idx
    ov::ParameterVector params_to_remove;
    std::set<std::size_t> weights_to_unpack;

    std::vector<ov::Tensor> zero_points;  // zero points for closures, if needed
};

ClosureRemap build_remap(const Function& fbody, const DCOFFParams& p);
void apply_remap(Subgraph& fcall, const ClosureRemap& m);
void finalize_remap(Function& fbody, Subgraph& fsg, const ClosureRemap& m);

// Various patterns here

namespace SymmNoZP {

class DCOFFPassBase : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmNoZP::DCOFFPassBase");

protected:
    DCOffMode m_dcoff_mode = DCOffMode::CAST_ONLY;
    ov::element::Type m_dcoff_type;
    DCOFFParamRef m_params_to;

    std::shared_ptr<ov::Node> paramA, paramB, toFP32, mulply, cvtopt;
    bool matcher_callback(ov::pass::pattern::Matcher& m);

public:
    DCOFFPassBase(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);

    virtual void build();
    virtual void reconnect_root_to_convert(ov::pass::pattern::Matcher& m) = 0;
};

class DCOFFPassMatMul final : public DCOFFPassBase {
    std::shared_ptr<ov::Node> matmul;

public:
    using DCOFFPassBase::DCOFFPassBase;
    void build() override;
    void reconnect_root_to_convert(ov::pass::pattern::Matcher& m) override;
};

class DCOFFPassGather final : public DCOFFPassBase {
    std::shared_ptr<ov::Node> gather;

public:
    using DCOFFPassBase::DCOFFPassBase;
    void build() override;
    void reconnect_root_to_convert(ov::pass::pattern::Matcher& m) override;
};
// FIXME: The above two can probably be replaced with a more genering pattern
// (ending at Multiply but with more connection constraints)

}  // namespace SymmNoZP

namespace SymmZP {  // TODO: Not sure if it is actually Symm..

class DCOFFPassBase : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::DCOFFPassBase");

protected:
    DCOffMode m_dcoff_mode = DCOffMode::CAST_ONLY;
    ov::element::Type m_dcoff_type;
    DCOFFParamRef m_params_to;

    std::shared_ptr<ov::Node> paramA, constB, paramC, cvtA, cvtB, subtr, mulply;
    bool matcher_callback(ov::pass::pattern::Matcher& m);

public:
    DCOFFPassBase(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);

    virtual void build();
    virtual void reconnect_root(ov::pass::pattern::Matcher& m) = 0;
};

class DCOFFPassReshape1 final : public DCOFFPassBase {
    std::shared_ptr<ov::Node> reshpe;

public:
    using DCOFFPassBase::DCOFFPassBase;
    void build() override;
    void reconnect_root(ov::pass::pattern::Matcher& m) override;
};

class DCOFFPassConvert1 final : public DCOFFPassBase {
    std::shared_ptr<ov::Node> cvtEnd;

public:
    using DCOFFPassBase::DCOFFPassBase;
    void build() override;
    void reconnect_root(ov::pass::pattern::Matcher& m) override;
};

class DCOFFPassReshape2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::DCOFFPassReshape2");
    DCOFFPassReshape2(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);
};

class DCOFFPassReshape3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::DCOFFPassReshape3");
    DCOFFPassReshape3(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);
};

class DCOFFPassReshape4 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::DCOFFPassReshape4");
    DCOFFPassReshape4(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);
};

class CWAI1 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::CWAI1");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    explicit CWAI1(Results scales);
};

class CWAI2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::CWAI2");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    explicit CWAI2(Results scales);
};

class CWAI3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::SymmZP::CWAI3");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    explicit CWAI3(Results scales);
};

}  // namespace SymmZP

namespace AsymmZP {
class DCOFFPassReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::AsymmZP::DCOFFPassReshape");
    DCOFFPassReshape(DCOffMode dcoff_mode, ov::element::Type dcoff_type, DCOFFParamRef pref);
};

}  // namespace AsymmZP

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
