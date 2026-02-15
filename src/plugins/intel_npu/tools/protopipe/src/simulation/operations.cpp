//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/operations.hpp"
#include "utils/error.hpp"

cv::GProtoArgs InferCall::operator()(const cv::GProtoArgs& inputs) {
    cv::GInferInputs infer_inputs;
    for (int i = 0; i < inputs.size(); ++i) {
        auto gmat = cv::util::get<cv::GMat>(inputs[i]);
        infer_inputs[input_names[i]] = gmat;
    }
    auto infer_outputs = cv::gapi::infer(tag, infer_inputs);
    cv::GProtoArgs outputs;
    for (int i = 0; i < output_names.size(); ++i) {
        outputs.emplace_back(infer_outputs.at(output_names[i]));
    }
    return outputs;
}

std::vector<cv::GMat> GDummyM::on(const std::vector<cv::GMat>& ins, const uint64_t delay_in_us,
                                  const std::vector<IDataProvider::Ptr>& providers, const bool disable_copy) {
    std::vector<cv::GShape> shapes;
    std::vector<cv::detail::OpaqueKind> op_kinds;
    std::vector<cv::detail::HostCtor> host_ctors;
    std::vector<cv::GArg> gargs;
    std::vector<cv::detail::OpaqueKind> out_kinds;

    gargs.emplace_back(providers);
    gargs.emplace_back(delay_in_us);
    gargs.emplace_back(disable_copy);

    for (int i = 0; i < ins.size(); ++i) {
        auto shape = cv::detail::GTypeTraits<cv::GMat>::shape;
        shapes.push_back(shape);
        auto op_kind = cv::detail::GTypeTraits<cv::GMat>::op_kind;
        op_kinds.push_back(op_kind);
        host_ctors.push_back(cv::detail::GObtainCtor<cv::GMat>::get());
        gargs.emplace_back(ins[i]);
    }

    const size_t num_outputs = providers.size();
    for (int i = 0; i < num_outputs; ++i) {
        auto op_kind = cv::detail::GTypeTraits<cv::GMat>::op_kind;
        out_kinds.push_back(op_kind);
    }

    using namespace std::placeholders;
    cv::GKernel k{GDummyM::id(),
                  "",
                  std::bind(&GDummyM::getOutMeta, _1, _2),
                  std::move(shapes),
                  std::move(op_kinds),
                  std::move(host_ctors),
                  std::move(out_kinds)};

    cv::GCall call(std::move(k));
    call.setArgs(std::move(gargs));

    std::vector<cv::GMat> outs;
    outs.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
        outs.push_back(call.yield(i));
    }

    return outs;
}

cv::GMetaArgs GDummyM::getOutMeta(const cv::GMetaArgs&, const cv::GArgs& args) {
    const auto& providers = args.front().get<std::vector<IDataProvider::Ptr>>();
    cv::GMetaArgs out_metas;
    out_metas.reserve(providers.size());
    for (auto provider : providers) {
        out_metas.emplace_back(provider->desc());
    }
    return out_metas;
}

cv::gapi::GBackend GCPUDummyM::backend() {
    return cv::gapi::cpu::backend();
}

cv::GCPUKernel GCPUDummyM::kernel() {
    return cv::GCPUKernel(&GCPUDummyM::call, &GCPUDummyM::setup);
}

void GCPUDummyM::setup(const cv::GMetaArgs& metas, cv::GArgs gargs, cv::GArg& state, const cv::GCompileArgs& args) {
    state = cv::GArg(std::make_shared<State>());
    auto providers = gargs.front().get<std::vector<IDataProvider::Ptr>>();
    for (auto& provider : providers) {
        provider->reset();
    }
}

void GCPUDummyM::call(cv::GCPUContext& ctx) {
    using namespace std::chrono;
    const bool disable_copy = ctx.inArg<bool>(2u);
    uint64_t elapsed = disable_copy ? 0u : utils::measure<microseconds>([&]() {
        auto& providers = ctx.inArg<std::vector<IDataProvider::Ptr>>(0u);
        for (size_t i = 0; i < providers.size(); ++i) {
            providers[i]->pull(ctx.outMatR(static_cast<int>(i)));
        }
    });
    const auto delay_in_us = ctx.inArg<uint64_t>(1u);
    utils::busyWait(microseconds{std::max(delay_in_us - elapsed, uint64_t{0})});
}

cv::GProtoArgs DummyCall::operator()(const cv::GProtoArgs& inputs) {
    std::vector<cv::GMat> gmats;
    gmats.reserve(inputs.size());
    for (auto& in : inputs) {
        gmats.emplace_back(cv::util::get<cv::GMat>(in));
    }
    auto outputs = GDummyM::on(gmats, delay_in_us, providers, disable_copy);
    cv::GProtoArgs proto_outputs;
    for (auto& out : outputs) {
        proto_outputs.emplace_back(cv::GProtoArg{out});
    }
    return proto_outputs;
}

cv::GProtoArgs CompoundCall::operator()(const cv::GProtoArgs& inputs) {
    ASSERT(inputs.size() == 1)
    cv::GMat in = cv::util::get<cv::GMat>(inputs[0]);

    cv::GProtoArgs proto_outputs;
    proto_outputs.emplace_back(GCompound::on(in, function));
    return proto_outputs;
}
