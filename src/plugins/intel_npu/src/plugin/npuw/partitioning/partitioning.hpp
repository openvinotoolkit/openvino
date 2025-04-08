// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "../lazy_tensor.hpp"
#include "../spatial.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {

struct Subgraph {
    // Note: This vector only specifies Link types of parameters
    // - e.g. Parameters which are coming into this subraph from
    // other subgraphs or from the parent Model's Parameter vector
    // directly
    ov::ParameterVector _parameters;
    ov::ResultVector _results;
    ov::SinkVector _sinks;

    std::string _affinity;
    std::size_t _ops = 0u;
    float _gflops = 0.f;
    bool _optimized_out = false;

    std::string _avoid_list;
    std::string _tag;

    // Function calls only (note: all the above fields are not used)
    //
    // FIXME: Replace with variant or some other proper way (maybe
    // even a class hierarchy)
    std::string _repeated_id;  // FIXME: What's the difference
    std::string _funcall;      // ..between these two?
    std::vector<ov::Tensor> _closure;
    std::vector<ov::Tensor> _scales;  // Scale coeffs for manual unpacking
    std::vector<ov::Tensor> _zerops;  // Zero points for manual unpacking

    // Stores transformation history for weights which will be applied before inference
    std::vector<weights::LazyTensor> _lazy_closure;
    // FIXME: shouldn't be here. Needed to not unpack some lazy closures in DCOFF
    std::vector<bool> _is_lazy_unpack;

    bool _forced_to_fcall = false;

    struct Gather {
        // NB.: int64_t is strange but it is used by OV to refer to parameters
        int64_t dst_idx = -1;
        int64_t src_idx = -1;
        int64_t idx_idx = -1;
    };
    Gather _host_gather;

    using Ref = std::reference_wrapper<Subgraph>;
};

struct Function {
    std::shared_ptr<ov::Model> _model;
    std::size_t _param_offset;
    std::size_t _num_params_total;

    std::string _tag;  // derived from the partitioning

    // Mapping: from a prototype {Layer/input_idx} to {param_idx}
    // NOTE: it seems it is required only for `matchRepeatedSubgraphs()'
    std::map<std::pair<std::string, std::size_t>, std::size_t> _param_mapping;

    std::optional<ov::npuw::function::Spatial> _spatial;

    // FIXME: shouldn't be here. Needed to not unpack some lazy closures in DCOFF
    std::set<std::size_t> _idx_lazy_unpack;
};

struct Group {
    std::vector<std::string> input_layers;
    std::vector<std::string> output_layers;
    std::vector<std::string> all_layers;

    std::string repeated_id;
    float gflops;

    std::string avoid_list;
    std::string tag;

    // Set to true if the Group was forcibly turned to functon. Such
    // function has just a single associated funcall and are subjects
    // to some optimizations (simplifications).
    bool forced_to_fcall = false;

    ov::npuw::Subgraph sg;
};

struct RepeatedBlock {
    using MatchedLayers = std::set<std::string>;
    using MatchedBank = std::vector<MatchedLayers>;
    MatchedBank matches;
    MatchedBank consts;
    MatchedBank scalars;
};

struct Ensemble {
    float gflops;
    std::vector<Group> groups;

    // Just a map as I don't expect 100s of _different_
    // repeated blocks here
    std::map<std::string, RepeatedBlock> repeated;
};

using LinkTo = std::pair<size_t /*submodel_idx*/
                         ,
                         size_t /*param_idx*/
                         >;
using LinkFrom = std::pair<size_t /*submodel_idx*/
                           ,
                           size_t /*result_idx*/
                           >;
using Links = std::map<LinkTo, LinkFrom>;

struct Partitioning {
    std::vector<Subgraph> subgraphs;
    Links input_to_prev_output;

    // Function: A proper name of a repeated block
    std::map<std::string, Function> functions;

    std::size_t total_ops = 0u;
    float total_gflops = 0.f;
};

Partitioning getPartitioning(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& config);

}  // namespace npuw
}  // namespace ov
