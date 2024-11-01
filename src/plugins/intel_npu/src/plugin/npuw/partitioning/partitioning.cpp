// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning.hpp"

#include <memory>

#include "../logging.hpp"
#include "../util.hpp"
#include "intel_npu/config/npuw.hpp"
#include "online/compiler.hpp"
#include "online/utils/utils.hpp"  // getMetaDesc
#include "openvino/core/parallel.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "patterns/dcoff.hpp"
#include "patterns/opt.hpp"

namespace ov {
namespace npuw {
inline bool operator==(const std::reference_wrapper<Subgraph>& lhs, const std::reference_wrapper<Subgraph>& rhs) {
    ov::npuw::Subgraph& llink = lhs.get();
    ov::npuw::Subgraph& rlink = rhs.get();
    return &llink == &rlink;
}
}  // namespace npuw
}  // namespace ov

template <typename T2>
struct std::hash<std::pair<ov::npuw::Subgraph::Ref, T2>> {
    std::size_t operator()(std::pair<ov::npuw::Subgraph::Ref, T2> const& p) const noexcept {
        ov::npuw::Subgraph& sg = p.first.get();
        std::size_t h1 = std::hash<void*>{}(&sg);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

namespace {

class FuncallEverywhere {
    std::shared_ptr<ov::Model> m_model;
    std::set<std::string> m_known;
    std::size_t m_new_id = 0u;

    mutable bool m_enabled = false;
    mutable std::once_flag m_once;

    ::intel_npu::Config& m_cfg;

public:
    explicit FuncallEverywhere(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg)
        : m_model(model),
          m_cfg(cfg) {}

    void register_known(const std::string fcn_id) {
        m_known.insert(fcn_id);
    }

    std::string register_new() {
        std::string new_fcn = "FCEW" + ov::npuw::util::fmt(m_new_id++, 100);

        // Just make an assert here: assume we don't need to handle
        // collisions
        NPUW_ASSERT(m_known.count(new_fcn) == 0);
        return new_fcn;
    }

    bool enabled() const {
        std::call_once(m_once, [&]() {
            const bool fce_opt = m_cfg.get<::intel_npu::NPUW_FUNCALL_FOR_ALL>();
            if (fce_opt) {
                LOG_WARN("Every subgraph in " << m_model->get_friendly_name()
                                              << " will be turned to a function: may cause performance issues");
                m_enabled = true;
            }
        });
        return m_enabled;
    }
};

struct BankContains {
    std::string name;
    bool operator()(const ov::npuw::RepeatedBlock::MatchedLayers& lrs) {
        return lrs.count(name) > 0;
    }
};

struct ProducesResult {
    bool operator()(const std::shared_ptr<ov::Node>& node) {
        std::set<ov::Input<ov::Node>> all_readers;
        for (auto&& out : node->outputs()) {
            const auto& these_readers = out.get_target_inputs();
            all_readers.insert(these_readers.begin(), these_readers.end());
        }
        return std::any_of(all_readers.begin(), all_readers.end(), [](const ov::Input<ov::Node>& iport) {
            return ov::op::util::is_output(iport.get_node());
        });
    }
};

ov::npuw::Ensemble load_groups(const std::shared_ptr<ov::Model>& model, const std::string& path_to_plan) {
    // Try to load the partitioning plan...
    NPUW_ASSERT(!path_to_plan.empty());

    std::ifstream ifs(path_to_plan);
    if (!ifs) {
        LOG_ERROR("Couldn't open " << ::intel_npu::NPUW_PLAN().key() << " pointing to " << path_to_plan << "!");
        return {};
    }

    // If file is specified and it exists, try to parse the XML document
    LOG_INFO("Loading plan from " << path_to_plan << "...");
    pugi::xml_document xml_doc;
    pugi::xml_parse_result xml_res = xml_doc.load_file(path_to_plan.c_str());
    if (xml_res.status != pugi::status_ok) {
        LOG_ERROR("Couldn't parse the plan XML!");
        return {};
    }

    // Load partitioning structure from the XML document.
    // Note the .sg part is not filled at this point

    using namespace ov::util::pugixml;
    auto root = xml_doc.document_element();

    // Load groups first
    // clang-format off
    std::vector<ov::npuw::Group> partitions;
    auto groups = root.child("partitioning");
    FOREACH_CHILD(group, groups, "group") {
        partitions.push_back(ov::npuw::Group{});
        ov::npuw::Group& this_group = partitions.back();
        this_group.gflops = get_float_attr(group, "gflops");
        this_group.repeated_id = get_str_attr(group, "repeated", "");
        this_group.avoid_list = get_str_attr(group, "avoid", "");
        this_group.tag = get_str_attr(group, "tag", "");
        FOREACH_CHILD(input, group, "input") {
            this_group.input_layers.push_back(get_str_attr(input, "name"));
        }
        FOREACH_CHILD(output, group, "output") {
            this_group.output_layers.push_back(get_str_attr(output, "name"));
        }
        FOREACH_CHILD(layer, group, "layer") {
            this_group.all_layers.push_back(get_str_attr(layer, "name"));
        }
    }

    // Load repeating blocks, if any
    std::map<std::string, ov::npuw::RepeatedBlock> repeated;
    auto reps = root.child("repeated");
    if (reps) {
        FOREACH_CHILD(block, reps, "block") {
            ov::npuw::RepeatedBlock this_block;
            FOREACH_CHILD(match, block, "match") {
                ov::npuw::RepeatedBlock::MatchedLayers mls;
                FOREACH_CHILD(layer, match, "layer") {
                    mls.insert(get_str_attr(layer, "name"));
                }
                this_block.matches.push_back(std::move(mls));
            }  // match
            repeated[get_str_attr(block, "id")] = std::move(this_block);
        }  // block
    }      // if(reps)
    // clang-format on

    LOG_INFO("Found " << repeated.size() << " different repeated block(s)");

    return ov::npuw::Ensemble{get_float_attr(root, "gflops"), std::move(partitions), std::move(repeated)};
}

class Partitioner {
private:
    // External state - taken in the consturctor
    const std::shared_ptr<ov::Model>& model;
    ov::npuw::Ensemble& ens;
    ov::npuw::Partitioning& P;

    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    using RPtr = std::shared_ptr<ov::op::v0::Result>;
    using SubgParam = std::pair<ov::npuw::Subgraph::Ref, PPtr>;
    using SubgResult = std::pair<ov::npuw::Subgraph::Ref, RPtr>;
    using LinkPtrTo = std::pair<size_t /*submodel_idx*/
                                ,
                                PPtr /*param ptr*/
                                >;
    using LinkPtrFrom = std::pair<size_t /*submodel_idx*/
                                  ,
                                  RPtr /*result ptr*/
                                  >;
    using LinkPtrs = std::map<LinkPtrTo, LinkPtrFrom>;
    LinkPtrs subgraph_ptr_links;

    // Internal state - not exposed anywhere
    struct FunctionPipeline {
        using VM = std::vector<std::shared_ptr<ov::Model>>;
        using VS = std::vector<ov::npuw::Subgraph::Ref>;
        VM mdls;
        VS refs;
        std::unordered_set<std::shared_ptr<ov::Node>> consts_to_keep;

        // Map every function call instance' Parameter and result
        // back to its prototype Parameter and Result
        std::unordered_map<SubgParam, PPtr> param_call_to_proto;
        std::unordered_map<SubgResult, RPtr> result_call_to_proto;
    };
    std::map<std::string, FunctionPipeline> all_functions;

    // NB(dm): Sort of warning here that this mapping is used
    // across all functions. Probably it would be easier and
    // more efficient to have one per function group
    std::unordered_map<std::string, std::string> layer_to_prototype;

    // Some of the scalars might be shared by different groups within
    // the same repeated block family. We need to keep them counted for sanity checks.
    // Matches a pair of {func_name, layer_name} to it's counter.
    std::map<std::pair<std::string, std::string>, size_t> dup_scalars;

    using Match = std::function<bool(const std::shared_ptr<ov::Node>& node)>;
    void propagate(const std::string& func_name, const Match& test, ov::npuw::RepeatedBlock::MatchedBank& bank);

    void createFunction(FunctionPipeline& func_ggg);

    // NB(dm): This method should get a better place, it is here only because
    // it is tied to the Function structure (but, in fact, not so much)
    void identifySpatialRange(ov::npuw::Function& f);

    template <typename T, typename M>
    void rearrange_to_function_protocol(ov::npuw::Subgraph::Ref func_ref,
                                        const std::vector<T>& protocol,
                                        std::vector<T>& call,
                                        const M& call_to_proto,
                                        bool required = false) {
        LOG_DEBUG("Rearranging...");
        LOG_BLOCK();
        LOG_DEBUG("Protocol: " << protocol.size());
        for (auto&& p : protocol) {
            LOG_BLOCK();
            LOG_DEBUG(p);
        }
        std::vector<T> to_proto;
        LOG_DEBUG("Call: " << call.size());
        for (auto&& c : call) {
            LOG_BLOCK();
            auto p_c = call_to_proto.at(typename M::key_type(func_ref, c));
            to_proto.push_back(p_c);
            LOG_DEBUG(c << " (which is " << p_c << ")");
        }
        std::vector<T> call_tmp = call;
        call = std::vector<T>(protocol.size(), nullptr);
        for (std::size_t i = 0; i < protocol.size(); i++) {
            auto iter = std::find(to_proto.begin(), to_proto.end(), protocol[i]);
            if (iter != to_proto.end()) {
                // There was an example in LLama where the last (32th) repeated
                // block had 3 outputs instead of 4 with several partitioning models.
                std::size_t j = std::distance(to_proto.begin(), iter);
                LOG_DEBUG("Put " << j << " element to " << i << " as in the protocol");
                call[i] = call_tmp[j];
            } else if (required) {
                OPENVINO_THROW("NPUW: Parameter from protocol is not found in the call!");
            }
        }
    }

    // FIXME: a fix to overcome the model with duplicate friendly names in constants
    std::string get_unique_name(const std::shared_ptr<ov::Node> node_ptr) {
        if (!node_ptr) {
            OPENVINO_THROW("NPUW: Fatal error");
        }
        if (!ov::is_type<ov::op::v0::Constant>(node_ptr)) {
            OPENVINO_THROW("NPUW: trying to get a unique name of a non-Constant node");
        }
        // FIXME: cache this
        return node_ptr->get_friendly_name() + " with meta " + ov::npuw::online::util::getMetaDesc(node_ptr) +
               " with output " + (*node_ptr->output(0).get_target_inputs().begin()).get_node()->description();
    }

public:
    Partitioner(const std::shared_ptr<ov::Model>& _model,
                ov::npuw::Ensemble& _ens,
                ov::npuw::Partitioning& _P,
                ::intel_npu::Config& _cfg)
        : model(_model),
          ens(_ens),
          P(_P),
          func_pipeline_type(FunctionPipelineType::FOLD),
          cfg(_cfg) {}

    ////////////////////////////////////////////////////////
    // Partitioning execution pipeline

    // Partitioning to subgraph mapping
    void identifySubgraphs();

    // Function folding subroutines
    enum class FunctionPipelineType { FOLD, CWAI };
    std::vector<std::string> initFunctionPipeline(FunctionPipelineType utype);

    // this implicit shared state can be lifted to some new
    // kind of object
    void propagateSlices(const std::string& func_name);
    void propagateConverts(const std::string& func_name);
    void propagateWeights(const std::string& func_name);
    void propagateScalars(const std::string& func_name);
    void propagateConvertsOut(const std::string& func_name);
    void sanityCheck(const std::string& func_name);
    void saveTinyConstants(const std::string& func_name);
    void saveScaleFactors(const std::string& func_name);
    void saveRepeatedConstants(const std::string& func_name);
    void matchParameters(const std::string& func_name);
    void matchResults(const std::string& func_name);
    void createFunction(const std::string& func_name);
    void matchRepeatedSubgraphs(const std::string& func_name);
    void spatial(const std::string& func_name);
    void optimize(const std::string& func_name);
    void decompressionCutOff(const std::string& func_name);

    // Final steps
    void finalizeLinks();

private:
    FunctionPipelineType func_pipeline_type;
    ::intel_npu::Config& cfg;
};

void Partitioner::identifySubgraphs() {
    LOG_INFO("Identifying subgraphs for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    using namespace ov::npuw;
    std::vector<ov::npuw::Group>& partitions = ens.groups;

    // Apply partitioning changes to the original model
    // but first cache all nodes to identify by name
    using NodeSPtr = std::shared_ptr<ov::Node>;
    std::unordered_map<NodeSPtr, LinkPtrFrom> result_cache;
    std::unordered_map<std::string, NodeSPtr> node_id_cache;
    for (auto&& node_ptr : model->get_ordered_ops()) {
        node_id_cache[node_ptr->get_friendly_name()] = node_ptr;
    }
    LOG_INFO("Caching done: " << node_id_cache.size() << " layers.");

    // Accumulate knowledge about known OV layers when walking
    // over a topologically-sorted list.
    std::unordered_set<NodeSPtr> nodes_known_now;

    // FIXME: Need to do some sanity checks here. What if partitioning
    // has been generated for another variation of this model?
    // What if that was a completely different model?
    std::size_t this_group_idx = 0u;  // FIXME: Replace with indexed()
    for (auto&& group : partitions) {
        LOG_INFO("Process partition " << this_group_idx << "...");
        LOG_BLOCK();
        // Sanity check: all group's layers present in the model
        std::unordered_set<NodeSPtr> group_nodes;
        for (auto&& layer : group.all_layers) {
            auto it = node_id_cache.find(layer);
            if (node_id_cache.end() == it) {
                OPENVINO_THROW("NPUW: Fatal error - partitition refers to layer \"",
                               layer,
                               "\" which was not found in the ov::Model");
            }
            group_nodes.insert(it->second);
        }
        group.sg._repeated_id = group.repeated_id;
        group.sg._forced_to_fcall = group.forced_to_fcall;
        group.sg._gflops = group.gflops;
        group.sg._ops = group.all_layers.size();
        P.total_ops += group.sg._ops;

        group.sg._avoid_list = group.avoid_list;
        group.sg._tag = group.tag;
        // Note inputs and outputs are included in the above set, so if
        // we are here, those nodes should be present in the model.

        // In the subgraph, every "input" layer is the first (top) operation
        // in the submodel (meaning it only depends on Parameters and Constants).
        // In the original model, though, this layer is connected to smt.
        // OpenVINO edge model is following:
        // SomeNode:#Output -> #Input:ThisNode.
        // In order to properly maintain connectivity of the partitioned model,
        // we need to track these connections here.

        // FIXME: But for now, just break all the connections!
        // Note: input/output layers are _parts of the group_, they are like
        // beginning/end of the subgraph.

        // Input layers may be connected to the same producer nodes, weights,
        // or parameters. Cache those to avoid duplicating the parameters.
        std::unordered_map<NodeSPtr, NodeSPtr> input_mapping;

        // In several cases a model can be slightly altered after the partitioning
        // plan was done. E.g., new slices or converts may be added on inputs/
        // outputs. Add a special handling for this case.
        std::unordered_set<NodeSPtr> extra_params;
        auto parameter_as_is = [&input_mapping](NodeSPtr orig_node) {
            auto it = input_mapping.find(orig_node);
            if (it != input_mapping.end()) {
                return it->second;
            }
            input_mapping[orig_node] = orig_node;
            return orig_node;
        };
        auto parameter_from = [&input_mapping](ov::Output<ov::Node> output) {
            auto orig_node = output.get_node_shared_ptr();
            auto it = input_mapping.find(orig_node);
            if (it != input_mapping.end()) {
                return it->second;
            }

            // In some weird combinations we can substitute a run-time parameter
            // with a Constant. Normally this happens on ShapeOf/... subgraphs.
            // In this case, adding a new parameter in a subnode will only break
            // stuff as OpenVINO will be unable to calculate some output tensor
            // shapes without tricks. So...
            NodeSPtr result;
            auto& output_tensor = output.get_tensor();
            if (output_tensor.has_and_set_bound()) {
                // if has_and_set_bound() == true, lower/upper values are the same tensor.
                auto new_const = std::make_shared<ov::op::v0::Constant>(output_tensor.get_upper_value());
                result = std::static_pointer_cast<ov::Node>(new_const);
                LOG_VERB("Found bound value in " << output << ", substituting it with " << new_const);
            } else {
                // OK, actually introduce a parameter, cache it, and return.
                auto new_param =
                    std::make_shared<ov::op::v0::Parameter>(output.get_element_type(), output.get_partial_shape());
                result = std::static_pointer_cast<ov::Node>(new_param);
            }
            input_mapping[orig_node] = result;
            return result;
        };
        for (auto&& input_layer_name : group.input_layers) {
            LOG_VERB("Processing group's input layer " << input_layer_name);
            auto input_layer_ptr = node_id_cache.at(input_layer_name);
            if (input_layer_ptr->inputs().empty()) {
                OPENVINO_THROW("The group's input layer ",
                               input_layer_name,
                               " has NO INPUTS!! - Graph contracts are broken??");
            }
            for (auto&& input_desc : input_layer_ptr->inputs()) {
                LOG_BLOCK();
                const auto input_node = input_desc.get_source_output().get_node_shared_ptr();
                LOG_DEBUG("Checking " << input_node);

                if (ov::op::util::is_parameter(input_node)) {
                    // Input to this subgraph layer is already a Parameter (original graph
                    // input). Track it among the subgraph parameters and continue.
                    parameter_as_is(input_node);
                } else if (ov::op::util::is_constant(input_node)) {
                    // Input to this subgraph layer is Const (weight). Don't do anything here.
                    continue;
                } else if (ov::is_type<ov::op::v0::Convert>(input_node) &&
                           ov::op::util::is_constant(input_node->input(0).get_source_output().get_node_shared_ptr())) {
                    // "Just-an-op" case, popped here again
                    // FIXME: Finally introduce my own test routine for that!
                    // Don't do anything here too.
                    continue;
                } else if ((ov::is_type<ov::op::v8::Slice>(input_node) ||
                            ov::is_type<ov::op::v0::Convert>(input_node)) &&
                           !nodes_known_now.count(input_node) &&
                           ov::op::util::is_parameter(input_node->input(0).get_source_output().get_node_shared_ptr())) {
                    // So the situation is:
                    //  - a group has an input layer
                    //  - which reads from a Slice or Convert
                    //  - which reads from a Parameter
                    //  - not a part of any prior group
                    // This happens when an offline plan is used with a kvcache
                    // model extended with slices to maintain zero-copy (LLM case)
                    auto extra_param = input_node->input(0).get_source_output().get_node_shared_ptr();
                    input_mapping[input_node] = extra_param;
                    extra_params.insert(extra_param);
                    LOG_DEBUG("Registered extra param " << extra_param);
                } else {
                    // Ok, this input is connected to some other node's output
                    // Replace this connection with a link to a newly created Parameter
                    // if this connection isn't coming from the same group like in pic below:
                    //
                    //  [par1]     [par2]
                    //   v          :
                    //   op1        :
                    //   : .........'
                    //   v v
                    //   op2
                    //   v
                    //   op3
                    // In this case both op1 and op2 will be marked input layers, but it doesn't
                    // meen there's nothing _before_ an input layer in the graph
                    if (group_nodes.find(input_node) == group_nodes.end()) {
                        // Can't use input_node here directly since parameter_from converts
                        // ov::Node to Output<Node> which some layers don't support by default.
                        auto new_param = parameter_from(input_desc.get_source_output());
                        ov::copy_runtime_info(input_node, new_param);
                        input_desc.replace_source_output(new_param);
                    }
                }  // if (is..)
            }      // for (inputs)
        }          // for (input_layers)
        // Transform the accumulated parameters to the subgraph model's input parameter vector
        // Also track the connectivity
        LOG_VERB("Populating _parameters...");
        group.sg._parameters.clear();

        // Stabilize input order - sort layers based on names
        using PairNodePtr = std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>;
        std::vector<PairNodePtr> input_mapping_sorted(input_mapping.begin(), input_mapping.end());
        std::sort(input_mapping_sorted.begin(),
                  input_mapping_sorted.end(),
                  [](const PairNodePtr& p1, const PairNodePtr& p2) {
                      // Sanity check
                      NPUW_ASSERT(p1.first->get_friendly_name() != p2.first->get_friendly_name());
                      return p1.first->get_friendly_name() < p2.first->get_friendly_name();
                  });

        // Now (after unknown slices/converts were introduced) params may be referred to
        // from multiple places in the model - so may be added multiple times to the
        // input mapping. This is a w/a, better they're added only once (TODO).
        // This set handles it.
        std::set<std::shared_ptr<ov::Node>> unique_params;
        for (auto&& im : input_mapping_sorted) {
            LOG_BLOCK();
            auto& src_node = im.first;
            auto& maybe_param = im.second;
            if (ov::op::util::is_parameter(maybe_param) && unique_params.count(maybe_param) == 0) {
                // some Parameters could fold into Constants, so only add real parameters
                auto this_param = std::static_pointer_cast<ov::op::v0::Parameter>(maybe_param);
                group.sg._parameters.push_back(this_param);
                unique_params.insert(maybe_param);
                if (src_node != this_param && extra_params.count(this_param) == 0) {
                    // Parameter node and the recorded src node are different
                    // so it is a cut-off point (see above, parameter_from()):
                    // - record connectivity between subgraphs.
                    // Exception: param is registered via slice or convert
                    const auto& link_from = result_cache.at(src_node);
                    const auto link_to = LinkPtrTo{this_group_idx, this_param};
                    subgraph_ptr_links[link_to] = link_from;
                }
            } else {
                // assert is_constant(), there's no other way
            }
        }  // for(input_mapping_sorted)

        // The same logic for group's final layers: replace their direct
        // connections with Result stubs (but remember where these outputs
        // were going to).
        LOG_VERB("Populating _results...");
        {
            // Before populating the output layers, do a quick Result->Output Layer
            // propagation to extend out output layers with the layers not mentioned
            // in the partitioning plan. This may happen if the plan was already exported,
            // but some changes were done to the model (like kvcache regrouping) after
            // that.
            // The idea is simple: walk over the group's all_layers and check if those
            // are producing results. If they are and they're not parts of the output_layers,
            // add them there.
            // Another case which is handled here is an extra Convert which can be
            // set as part of kvcache conversion routune.
            LOG_BLOCK();
            std::set<std::string> output_layers_cache(group.output_layers.begin(), group.output_layers.end());

            // Have to switch clang-format here to make cpplint happy
            // clang-format off

            for (auto&& op_name : group.all_layers) {
                auto layer_ptr = node_id_cache.at(op_name);
                if (ProducesResult {}(layer_ptr) && !output_layers_cache.count(op_name)) {
                    LOG_VERB("Adding " << op_name << " as an extra output layer since it is produces a Result");
                    output_layers_cache.insert(op_name);
                    group.output_layers.push_back(op_name);
                }
                for (auto&& oport : layer_ptr->outputs()) {
                    for (auto&& inport : oport.get_target_inputs()) {
                        auto reader_ptr = inport.get_node();
                        if (ov::is_type<ov::op::v0::Convert>(reader_ptr) &&
                            ProducesResult {}(reader_ptr->shared_from_this()) &&
                            !output_layers_cache.count(reader_ptr->get_friendly_name())) {
                            const auto& cvt_name = reader_ptr->get_friendly_name();
                            output_layers_cache.insert(cvt_name);
                            group.output_layers.push_back(cvt_name);
                        }
                    }
                }
            }  // for(all_layers)
            // clang-format on
        }
        std::size_t num_optimized_out_layers = 0u;
        for (auto&& output_layer_name : group.output_layers) {
            LOG_VERB("Processing group's output layer " << output_layer_name);
            LOG_BLOCK();
            auto output_layer_ptr = node_id_cache.at(output_layer_name);
            if (output_layer_ptr->outputs().empty()) {
                OPENVINO_THROW("The group's output layer ",
                               output_layer_name,
                               " has NO OUTPUTS!! - Graph contracts are broken??");
            }
            const auto old_results_size = group.sg._results.size();
            std::size_t num_optimized_out = 0;
            for (auto&& output_desc : output_layer_ptr->outputs()) {
                // NOT Every layer's output becomes a Result:
                // Some of those can be consumed by the same group, e.g.
                //
                //    op100
                //    v v v
                //    : : [res1]
                //    : v
                //    : op101
                //    v
                //    op102
                bool has_external_readers = false;
                NodeSPtr maybe_result = nullptr;
                auto readers = output_desc.get_target_inputs();
                // This is possible then some of layer's outputs are not used in the model.
                if (readers.empty()) {
                    LOG_VERB("Output layer " << output_desc.get_node()->get_friendly_name()
                                             << " was OPTIMIZED OUT since it has no readers.");
                    num_optimized_out++;
                    continue;
                }
                for (auto&& r : readers) {
                    // First, remember the connections (to replicate those
                    // at the npuw::CompiledModel level)
                    auto reader_node_ptr = r.get_node()->shared_from_this();
                    if (ov::op::util::is_output(reader_node_ptr)) {
                        maybe_result = std::move(reader_node_ptr);
                    } else if (group_nodes.find(reader_node_ptr) == group_nodes.end()) {
                        has_external_readers = true;
                    }
                }
                if (maybe_result) {
                    // This layer's output was connected to Result already.
                    // It happens when this layer is the original model's output
                    // Keep it to make the ugly top-level I/O matching procedure work.
                    // FIXME: This needs to be refactored
                    group.sg._results.push_back(std::dynamic_pointer_cast<ov::op::v0::Result>(maybe_result));
                    result_cache[output_layer_ptr] =
                        LinkPtrFrom{this_group_idx, std::dynamic_pointer_cast<ov::op::v0::Result>(maybe_result)};
                } else if (has_external_readers) {
                    // Introduce and record a new Result
                    // As the graph is processed in the topological order,
                    // details recorded about this output at this point will
                    // be handled as input details for some other subgraph.

                    if (output_desc.get_tensor().has_and_set_bound()) {
                        // Note: There is an optimization above where some Parameters
                        // (down-stream) can be folded to Constants. Same applies to
                        // Results which are inputs to these parameters. If the value is
                        // foldable, no need to introduce Result too.
                        // Actually it is OK for CPU and GPU, but it breaks the NPU plugin.
                        num_optimized_out++;
                        LOG_VERB("Discarding " << output_desc << " -- optimized out!");
                    } else {
                        auto new_result = std::make_shared<ov::op::v0::Result>(output_desc);
                        result_cache[output_layer_ptr] = LinkPtrFrom{this_group_idx, new_result};

                        ov::copy_runtime_info(output_desc.get_node_shared_ptr(), new_result);
                        group.sg._results.push_back(new_result);
                    }
                }
            }  // for (outputs)
            if (num_optimized_out == output_layer_ptr->outputs().size()) {
                num_optimized_out_layers++;
            }
            const auto new_results_size = group.sg._results.size();
            LOG_VERB("Note: Processing the group " << this_group_idx << " output layer " << output_layer_name
                                                   << " added " << new_results_size - old_results_size
                                                   << " new Result node(s)");
        }  // for (output_layers)
        if (group.sg._results.empty()) {
            if (num_optimized_out_layers != group.output_layers.size()) {
                OPENVINO_THROW("NPUW Fatal: No Results registered for group ", this_group_idx);
            } else {
                LOG_VERB("Completely optimize out group " << this_group_idx);
                group.sg._optimized_out = true;
            }
        }
        this_group_idx++;  // FIXME: indexed() is better!
        nodes_known_now.insert(group_nodes.begin(), group_nodes.end());
    }  // for (partitions)

    // Return what we've got here
    std::vector<Subgraph>& result = P.subgraphs;
    result.reserve(partitions.size());
    for (auto&& group : partitions) {
        result.push_back(std::move(group.sg));
    }
}

std::vector<std::string> Partitioner::initFunctionPipeline(FunctionPipelineType utype) {
    func_pipeline_type = utype;

    // Collect all groups of function call(s) and process them in groups
    std::map<std::string, int> idx;
    for (auto&& part_sg : P.subgraphs) {
        if (!part_sg._repeated_id.empty()) {
            auto pfix = "__" + std::to_string(idx[part_sg._repeated_id]++);
            const auto& fcid = func_pipeline_type == FunctionPipelineType::FOLD
                                   ? part_sg._repeated_id          // with folding, functions of the
                                                                   // same group have the same id
                                   : part_sg._repeated_id + pfix;  // with CWAI (which is not checked here)
                                                                   // every function gets its own id
            auto& u = all_functions[fcid];
            u.refs.push_back(std::ref(part_sg));
            u.mdls.push_back(
                std::make_shared<ov::Model>(part_sg._results,
                                            part_sg._sinks,
                                            part_sg._parameters,
                                            model->get_friendly_name() + "_" + part_sg._repeated_id + pfix));
        }
    }

    if (func_pipeline_type == FunctionPipelineType::CWAI) {
        // Early return - don't do anything else here.
        // Also update repeated_ids to uniques
        std::vector<std::string> functions;
        for (auto&& p : all_functions) {
            functions.push_back(p.first);
            p.second.refs.front().get()._repeated_id = p.first;
        }
        return functions;
    }

    // Then, populate a list of functions and the early layer_to_prototype mapping.
    // The latter will be refined in the following passes.
    LOG_VERB("Initialize layer banks...");
    std::vector<std::string> functions;
    for (auto&& p : all_functions) {
        LOG_BLOCK();
        functions.push_back(p.first);
        LOG_VERB("Processing function group " << p.first);

        auto& rep_block = ens.repeated.at(p.first);

        LOG_DEBUG("Use " << p.second.mdls.front()->get_friendly_name() << " as a template...");
        for (auto&& node_ptr : p.second.mdls.front()->get_ordered_ops()) {
            const auto& this_layer_name = node_ptr->get_friendly_name();
            auto layer_bank_iter =
                std::find_if(rep_block.matches.begin(), rep_block.matches.end(), BankContains{this_layer_name});
            if (layer_bank_iter != rep_block.matches.end()) {
                for (auto&& layer : *layer_bank_iter) {
                    LOG_BLOCK();
                    LOG_DEBUG(this_layer_name << " is a prototype of " << layer);
                    layer_to_prototype[layer] = this_layer_name;  // Link to self is ok
                }
            }
        }
    }
    LOG_VERB("Done");
    return functions;
}

void Partitioner::propagate(const std::string& func_name,
                            const Match& test,
                            ov::npuw::RepeatedBlock::MatchedBank& bank) {
    // NOTE: This routine assumes a Const is accessed by a single reader.
    // There may be situations where a Const (normally, not a Weight)
    // is accessed by multiple readers. This routine shouldn't be used
    // there.

    auto& model_group = all_functions.at(func_name).mdls;

    // PROTO writer (value) <OF> PROTO reader/port (key).
    // Readers are always registered in the bank (NOTE: _some_
    // bank, but probably not exactly the one which is passed
    // here as `bank' argument),
    // and Writers are subjects to this propagation process
    // (e.g. Convert! -> Op, Const! -> Convert)
    using ProtoReader = std::pair<std::string, size_t>;
    using ProtoReaders = std::set<ProtoReader>;
    auto dump_readers = [](const ProtoReaders& readers) {
        LOG_BLOCK();
        for (auto&& r : readers)
            LOG_DEBUG(r.first << " : " << r.second);
    };

    std::map<ProtoReaders, std::string> proto_writer_of;
    for (auto&& model : model_group) {
        LOG_DEBUG("Process function call " << model->get_friendly_name() << "...");
        LOG_BLOCK();

        // Cache model contents as sometimes Readers of Consts
        // may be outside. FIXME: Do only once when a tmp Model is created?
        std::unordered_set<ov::Node*> this_model_nodes;
        for (auto&& node_ptr : model->get_ordered_ops()) {
            this_model_nodes.insert(node_ptr.get());
        }

        for (auto&& node_ptr : model->get_ordered_ops()) {
            if (test(node_ptr)) {
                LOG_DEBUG("Process node " << node_ptr);
                const auto& this_layer_name = ov::is_type<ov::op::v0::Constant>(node_ptr)
                                                  ? get_unique_name(node_ptr)
                                                  : node_ptr->get_friendly_name();

                ProtoReaders this_node_readers, this_node_proto_readers;
                for (auto&& this_reader_iport : node_ptr->output(0).get_target_inputs()) {
                    LOG_BLOCK();
                    LOG_DEBUG("Read by " << this_reader_iport);
                    if (this_model_nodes.count(this_reader_iport.get_node()) == 0) {
                        LOG_BLOCK();
                        LOG_DEBUG("Link to the external reader (other submodel?) - skip");
                    } else {
                        this_node_readers.insert(ProtoReader{this_reader_iport.get_node()->get_friendly_name(),
                                                             this_reader_iport.get_index()});
                    }
                }  // for(this_reader_iport)
                LOG_DEBUG("Looking for proto accessess...");
                for (auto&& this_node_reader : this_node_readers) {
                    LOG_BLOCK();
                    LOG_DEBUG("Looking for proto of reader " << this_node_reader.first);
                    this_node_proto_readers.insert(
                        {layer_to_prototype.at(this_node_reader.first), this_node_reader.second});
                }
                auto bank_writer_iter = proto_writer_of.find(this_node_proto_readers);
                if (bank_writer_iter == proto_writer_of.end()) {
                    // FIXME: assert(bank_reader_name == proto(bank_reader_name))
                    // This is a first occasion for this Writer -> Reader pair.
                    // Register it for the further reference. Also add it to the layer bank
                    LOG_DEBUG("Register that " << this_layer_name << " is accessed by:");
                    dump_readers(this_node_proto_readers);
                    proto_writer_of[this_node_proto_readers] = this_layer_name;
                    layer_to_prototype[this_layer_name] = this_layer_name;
                    bank.push_back({this_layer_name});
                } else {
                    // Such occasion is already registered - find a suitable bank
                    // and add node there
                    const auto& this_writer_proto = bank_writer_iter->second;
                    auto suitable_bank_iter = std::find_if(bank.begin(), bank.end(), BankContains{this_writer_proto});
                    if (suitable_bank_iter == bank.end()) {
                        OPENVINO_THROW("Fatal: no suitable bank found");
                    }
                    // FIXME: add IF(DEBUG) to put the whole thing under condition
                    LOG_DEBUG("Register that " << this_layer_name << " is in fact " << this_writer_proto);
                    LOG_DEBUG("- As it is read by:");
                    dump_readers(this_node_readers);
                    LOG_DEBUG("- Which in turn are:");
                    dump_readers(this_node_proto_readers);
                    suitable_bank_iter->insert(this_layer_name);
                    layer_to_prototype[this_layer_name] = this_writer_proto;
                }  // if(iter==end)
            }      // test(node_ptr)
        }          // for(ordered_ops)
    }              // for(each)
}  // propagate

void Partitioner::propagateSlices(const std::string& func_name) {
    LOG_VERB("Propagate Slice nodes to matching banks for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // This is a special step. Normally we shouldn't do this here at all.
    // Extra slices may be added to the LLM kvcache models _after_
    // offline partitioning was done when adapting them to the zero
    // -copy kvcache path. In this case, we get extra slices in the model
    // which we need to add to the match banks

    auto& bank = ens.repeated.at(func_name).matches;
    auto match_fcn = [&](const std::shared_ptr<ov::Node>& node_ptr) -> bool {
        const auto& this_layer_name = node_ptr->get_friendly_name();
        return ov::is_type<ov::op::v8::Slice>(node_ptr) &&
                   bank.end() == std::find_if(bank.begin(), bank.end(), BankContains{this_layer_name})          // (0)
                   && ov::op::util::is_parameter(node_ptr->input(0).get_source_output().get_node_shared_ptr())  // (1)
                   && [&](std::set<ov::Input<ov::Node>>&& readers) -> bool {
            // FIXME: It could be all_of, but slices may have reads
            // outside of the function group (e.g., a single instance
            // of shape_of, taken only on the first kvcache tensor)
            return std::any_of(readers.begin(), readers.end(), [&](const ov::Input<ov::Node>& reader) -> bool {
                auto reply =
                    bank.end() !=
                    std::find_if(bank.begin(), bank.end(), BankContains{reader.get_node()->get_friendly_name()});
                return reply;
            });
        }(node_ptr->output(0).get_target_inputs());
    };
    propagate(func_name, match_fcn, bank);
    LOG_VERB("Done");
}

void Partitioner::propagateConverts(const std::string& func_name) {
    LOG_VERB("Propagate Convert nodes to matching banks for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // This is a special step. Normally we shouldn't do this here at all.
    // Here comes the original problem description:
    //
    //  ``Currently Ensemble omits Convert nodes from its displayable
    //    EnsOVIR graph model to simplify the visual representation.
    //    This causes a problem here - where in fact a Convert node is
    //    still part of the graph, but is NOT mentioned in the partitioning
    //    plan - and so, is missing in the matched layer banks. If we find
    //    one in the actual model we use as a function template, register
    //    a new bank for it - so every eligible Convert node in the function
    //    template registers its own bank. The eligibility is defined with
    //    the following rules:
    //    0. The node is missing in the matching bank
    //    1. The Convert node reads a Const (or Parameter, since recently)
    //    2. The Convert node has a sole consumer
    //    3. This sole consumer is present in the bank.''
    //
    // The propagation procedure is generic, but the matching isn't.
    auto& bank = ens.repeated.at(func_name).matches;
    auto match_fcn = [&](const std::shared_ptr<ov::Node>& node_ptr) -> bool {
        const auto& this_layer_name = node_ptr->get_friendly_name();
        if (!ov::is_type<ov::op::v0::Convert>(node_ptr)) {
            return false;
        }
        const auto& input_node_ptr = node_ptr->input(0).get_source_output().get_node_shared_ptr();
        return ov::is_type<ov::op::v0::Convert>(node_ptr) &&
               bank.end() == std::find_if(bank.begin(), bank.end(), BankContains{this_layer_name})  // (0)
               && (ov::op::util::is_constant(input_node_ptr) ||                                     // (1)
                   ov::op::util::is_parameter(input_node_ptr))                                      // (1)
               && node_ptr->output(0).get_target_inputs().size() == 1                               // (2)
               &&
               bank.end() !=
                   std::find_if(
                       bank.begin(),
                       bank.end(),
                       BankContains{
                           node_ptr->output(0).get_target_inputs().begin()->get_node()->get_friendly_name()});  // (3)
    };
    propagate(func_name, match_fcn, bank);
    LOG_VERB("Done");
}

void Partitioner::propagateWeights(const std::string& func_name) {
    LOG_VERB("Propagate Const Weights to matching banks for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // See explanation in propagateConverts().
    // The propagation procedure is generic, but the matching isn't.
    auto& const_bank = ens.repeated.at(func_name).consts;
    auto& layer_bank = ens.repeated.at(func_name).matches;
    auto match_fcn = [&](const std::shared_ptr<ov::Node>& node_ptr) -> bool {
        const auto& this_layer_name =
            ov::is_type<ov::op::v0::Constant>(node_ptr) ? get_unique_name(node_ptr) : node_ptr->get_friendly_name();
        return ov::is_type<ov::op::v0::Constant>(node_ptr) &&
               const_bank.end() == std::find_if(const_bank.begin(), const_bank.end(), BankContains{this_layer_name})
               // FIXME: workaround for scalars which might pass the weights check
               && (node_ptr->get_shape().size() > 1 ||
                   (node_ptr->get_shape().size() == 1 && node_ptr->get_shape()[0] > 10))
               // FIXME end
               && node_ptr->output(0).get_target_inputs().size() == 1 &&
               layer_bank.end() !=
                   std::find_if(
                       layer_bank.begin(),
                       layer_bank.end(),
                       BankContains{node_ptr->output(0).get_target_inputs().begin()->get_node()->get_friendly_name()});
    };
    propagate(func_name, match_fcn, const_bank);
    LOG_VERB("Done");
}

void Partitioner::propagateScalars(const std::string& func_name) {
    LOG_VERB("Propagate Const Scalars to matching banks for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // There are situations where the same Const operation is referred to
    // by multiple other Operations and so, in the repeated blocks model,
    // by multiple instances of the same (or many) function(s).

    // Such constants are safe to keep in-place (fused into the function
    // body, but first we need to make sure they have the same connectivity
    // from funcall to funcall (= add it to the matched bank).

    // the saveRepeatedConstants() will then take care of them.

    // See explanation in propagateConverts().
    // The propagation procedure is generic, but the matching isn't.
    auto& scalar_bank = ens.repeated.at(func_name).scalars;
    auto match_fcn = [&](const std::shared_ptr<ov::Node>& node_ptr) -> bool {
        const auto& this_layer_name =
            ov::is_type<ov::op::v0::Constant>(node_ptr) ? get_unique_name(node_ptr) : node_ptr->get_friendly_name();
        auto res =
            ov::is_type<ov::op::v0::Constant>(node_ptr) &&
            scalar_bank.end() == std::find_if(scalar_bank.begin(), scalar_bank.end(), BankContains{this_layer_name});
        if (ov::is_type<ov::op::v0::Constant>(node_ptr) &&
            scalar_bank.end() != std::find_if(scalar_bank.begin(), scalar_bank.end(), BankContains{this_layer_name})) {
            // FIXME: incorrect logic! This will also increment in case of multiple scalar outputs.
            // Instead it should only take shared scalars into account!
            dup_scalars[{func_name, this_layer_name}]++;
        }
        return res;
    };
    propagate(func_name, match_fcn, scalar_bank);

    LOG_VERB("Done");
}

void Partitioner::propagateConvertsOut(const std::string& func_name) {
    LOG_VERB("Propagate Converts on output nodes to match banks for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    using ProtoWriter = std::pair<std::string, size_t>;
    std::map<ProtoWriter, std::string> proto_reader_of;

    auto& model_group = all_functions.at(func_name).mdls;
    auto& bank = ens.repeated.at(func_name).matches;

    // Nodes we're looking for:
    // 1. Converts
    // 2. Missing in our match banks
    // 3. Its producer should be present in our match banks
    // 4. Standing in front of results
    auto test = [&](const std::shared_ptr<ov::Node>& node_ptr) {
        if (!ov::is_type<ov::op::v0::Convert>(node_ptr)) {  // 1
            return false;
        }
        const auto& this_layer_name = node_ptr->get_friendly_name();
        if (bank.end() != std::find_if(bank.begin(), bank.end(), BankContains{this_layer_name})) {  // 2
            return false;
        }
        const auto& in_layer_name = node_ptr->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name();
        if (bank.end() == std::find_if(bank.begin(), bank.end(), BankContains{in_layer_name})) {  // 3
            return false;
        }
        const auto& these_readers = node_ptr->output(0).get_target_inputs();
        return these_readers.size() == 1 &&
               ov::op::util::is_output(these_readers.begin()->get_node()->shared_from_this());  // 4
    };

    for (auto&& model : model_group) {
        LOG_DEBUG("Process function call " << model->get_friendly_name() << "...");
        LOG_BLOCK();

        for (auto&& node_ptr : model->get_ordered_ops()) {
            if (test(node_ptr)) {
                LOG_DEBUG("Process node " << node_ptr);
                const auto& this_layer_name = node_ptr->get_friendly_name();

                const auto& writer_out = node_ptr->input(0).get_source_output();
                {
                    LOG_BLOCK();
                    LOG_DEBUG("Written by " << writer_out);
                }
                ProtoWriter this_writer = {writer_out.get_node_shared_ptr()->get_friendly_name(),
                                           writer_out.get_index()};

                LOG_DEBUG("Looking for proto accessess...");
                ProtoWriter this_proto_writer = {layer_to_prototype.at(this_writer.first), this_writer.second};
                auto bank_writer_iter = proto_reader_of.find(this_proto_writer);
                if (bank_writer_iter == proto_reader_of.end()) {
                    // Register a new occasion
                    LOG_DEBUG("Register that " << this_layer_name << " is written by " << this_proto_writer.first
                                               << " : " << this_proto_writer.second);
                    proto_reader_of[this_proto_writer] = this_layer_name;
                    layer_to_prototype[this_layer_name] = this_layer_name;
                    bank.push_back({this_layer_name});
                } else {
                    // Find a suitable bank and find node there
                    const auto& this_reader_proto = bank_writer_iter->second;
                    auto suitable_bank_iter = std::find_if(bank.begin(), bank.end(), BankContains{this_reader_proto});
                    if (suitable_bank_iter == bank.end()) {
                        OPENVINO_THROW("Fatal: No suitable bank found");
                    }
                    LOG_DEBUG("Register that " << this_layer_name << " is in fact " << this_reader_proto);
                    LOG_DEBUG("- As it is written by:");
                    {
                        LOG_BLOCK();
                        LOG_DEBUG(this_writer.first << " : " << this_writer.second);
                    }
                    LOG_DEBUG("- Which in turn is:");
                    {
                        LOG_BLOCK();
                        LOG_DEBUG(this_proto_writer.first << " : " << this_proto_writer.second);
                    }
                    suitable_bank_iter->insert(this_layer_name);
                    layer_to_prototype[this_layer_name] = this_reader_proto;
                }
            }
        }  // for(ordered_ops)
    }

    LOG_VERB("Done");
}

void Partitioner::sanityCheck(const std::string& func_name) {
    LOG_VERB("Sanity check function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // All match banks should have the same size, and this size
    // is the # of functon calls in this group
    auto& rep_block = ens.repeated.at(func_name);
    auto& func_group = all_functions.at(func_name);
    LOG_DEBUG("The function has " << rep_block.matches.size() << " operation banks and " << rep_block.consts.size()
                                  << " constant banks");

    auto validate = [&func_group](const ov::npuw::RepeatedBlock::MatchedLayers& lrs) -> bool {
        for (auto&& l : lrs) {
            LOG_DEBUG(l);
        }
        if (lrs.size() != func_group.refs.size()) {
            LOG_WARN("Number of layers in match bank differs from # of function calls: " << lrs.size() << " != "
                                                                                         << func_group.refs.size());
            return false;
        }
        LOG_DEBUG("Validation passed");
        return true;
    };
    auto validate_scalars = [&](const ov::npuw::RepeatedBlock::MatchedLayers& lrs) -> bool {
        for (auto&& l : lrs) {
            LOG_DEBUG(l);
        }
        size_t f_dup_scalars = 0;
        for (const auto& l : lrs) {
            f_dup_scalars += dup_scalars[{func_name, l}];
        }
        if (lrs.size() != 1 && (lrs.size() + f_dup_scalars != func_group.refs.size()) &&
            (lrs.size() != func_group.refs.size())) {
            LOG_WARN("Number of layers in scalar match bank differs from 1 <OR> # of function calls "
                     << "<OR> # of function calls including duplicate scalars: " << lrs.size()
                     << " != " << func_group.refs.size() << " OR " << lrs.size() + f_dup_scalars
                     << " != " << func_group.refs.size());
            return false;
        }
        LOG_DEBUG("Validation passed");
        return true;
    };
    bool all_ok = true;
    for (auto& bank : rep_block.matches) {
        LOG_DEBUG("Validating operation bank...");
        LOG_BLOCK();
        all_ok &= validate(bank);
    }
    NPUW_ASSERT(all_ok);
    for (auto& bank : rep_block.consts) {
        LOG_DEBUG("Validating const bank...");
        LOG_BLOCK();
        all_ok &= validate(bank);
    }
    NPUW_ASSERT(all_ok);
    for (auto& bank : rep_block.scalars) {
        LOG_DEBUG("Validating scalar bank...");
        LOG_BLOCK();
        all_ok &= validate_scalars(bank);
    }
    NPUW_ASSERT(all_ok);
    // All Consts in all submodels should be registered at this point
    auto& consts = rep_block.consts;
    auto& scalars = rep_block.scalars;
    for (auto&& submodel : func_group.mdls) {
        LOG_DEBUG("Check " << submodel->get_friendly_name() << " constants...");
        LOG_BLOCK();

        std::unordered_set<ov::Node*> this_model_nodes;
        for (auto&& node : submodel->get_ordered_ops()) {
            this_model_nodes.insert(node.get());
        }

        for (auto&& node : submodel->get_ordered_ops()) {
            if (ov::op::util::is_constant(node) &&
                consts.end() == std::find_if(consts.begin(), consts.end(), BankContains{get_unique_name(node)}) &&
                scalars.end() == std::find_if(scalars.begin(), scalars.end(), BankContains{get_unique_name(node)})) {
                LOG_ERROR("Fatal: Const " << node->get_friendly_name() << "{ " << node->output(0) << " }"
                                          << " wasn't found in any bank");
                LOG_BLOCK();
                for (auto&& reader_input : node->output(0).get_target_inputs()) {
                    LOG_DEBUG("Accessed by "
                              << reader_input.get_node()->get_friendly_name() << " / " << reader_input.get_index()
                              << (this_model_nodes.count(reader_input.get_node()) > 0 ? " [internal]" : " [external]"));
                }
                all_ok = false;
            }
        }
    }
    NPUW_ASSERT(all_ok);
    LOG_VERB("Done");
}

void Partitioner::saveTinyConstants(const std::string& func_name) {
    // A simplified version of saveRepeatedConstants()

    LOG_VERB("Preserve tiny constants for " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    auto& func_group = all_functions.at(func_name);
    auto& model_group = func_group.mdls;

    using CT = ov::op::v0::Constant;

    for (auto&& op_node : model_group.front()->get_ordered_ops()) {
        for (auto&& iport : op_node->inputs()) {
            auto node = iport.get_source_output().get_node_shared_ptr();
            if (ov::op::util::is_constant(node)) {
                auto shape = node->output(0).get_shape();
                auto total =
                    std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
                if ((shape.size() == 0 || (shape.size() == 1 && shape[0] <= 10)) || (total <= 10)) {
                    LOG_DEBUG("[KEEP] It is safe to keep this bank in function");
                    func_group.consts_to_keep.insert(std::static_pointer_cast<CT>(node));
                } else {
                    LOG_DEBUG("[CUT ] This group of Const ops will be cut-off from the function");
                }
            }
        }
    }  // for(n)
    LOG_VERB("Done");
}

void Partitioner::saveScaleFactors(const std::string& func_name) {
    // A special step in the CWAI pipeline - mark the Scale
    // tensors to be preserved in the function bodies

    LOG_VERB("Preserve scale factors for " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    auto& func_group = all_functions.at(func_name);
    auto& model_group = func_group.mdls;

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    std::vector<CPtr> to_keep;

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::SymmZP::CWAI1>(std::ref(to_keep));
    rewr.add_matcher<ov::npuw::patterns::SymmZP::CWAI2>(std::ref(to_keep));
    rewr.add_matcher<ov::npuw::patterns::SymmZP::CWAI3>(std::ref(to_keep));
    rewr.run_on_model(model_group.front());

    for (auto&& const_to_keep : to_keep) {
        LOG_DEBUG("[KEEP] " << const_to_keep);
        func_group.consts_to_keep.insert(const_to_keep);
    }
    LOG_VERB("Done");
}

void Partitioner::saveRepeatedConstants(const std::string& func_name) {
    // Not all Constants can be cut off from the function body.  Some
    // of them are required for proper functioning and moving them to
    // parameters makes a static graph dynamic (e.g., when constant
    // defines some axis for Concat/Reshape/etc).  There may be a
    // chance that all these constants (in the same bank) have the
    // same value - in this case this value can be stored as a Const
    // in the function body.
    //
    // Checking all the Constants by value is overkill for FEIL, so
    // the following number of heuristics is used instead:
    //
    // To be saved, all constants in a bank must:
    // 0. Be of integer type
    // 1. Have 0 to 1 dimensions
    // 2. Have <= 10 elements in that single dimension, if any
    // 3. Have the equal values.
    //
    // The procedure below implements this.. but in a highly
    // inefficient manner If we start working with sptr<ov::Node>
    // objects directly (instead of their std::string identifiers),
    // the algorithm won't change much but will speed-up dramatically.

    LOG_VERB("Identify constants to save for function " << func_name << " in model " << model->get_friendly_name()
                                                        << "...");
    LOG_BLOCK();

    auto& func_group = all_functions.at(func_name);
    auto& model_group = func_group.mdls;
    auto& subgr_group = func_group.refs;
    auto& rep_block = ens.repeated.at(func_name);

    if (subgr_group.size() == 1) {
        // This is not a repeating block but a subgraph which is forced
        // to be function - run a simpler procedure here
        LOG_VERB("A FCEW function - switch to a simpler path...");
        LOG_BLOCK();
        saveTinyConstants(func_name);
        return;
    }

    // Build a cross-model <op name> to <op ptr> cache
    using CT = ov::op::v0::Constant;
    using CTPtr = std::shared_ptr<CT>;
    std::unordered_map<std::string, CTPtr> const_cache;
    for (auto&& m : model_group) {
        for (auto&& n : m->get_ordered_ops()) {
            if (ov::is_type<CT>(n)) {
                const_cache[get_unique_name(n)] = std::static_pointer_cast<CT>(n);
            }
        }
    }  // for(models)

    // Now walk through through every Const bank and inspect the above properties
    auto values_are_the_same = [](const CTPtr& node_a, const CTPtr& node_b) {
        switch (node_a->output(0).get_element_type()) {
#define HANDLE_CASE(t, T) \
    case ov::element::t:  \
        return node_a->get_vector<T>() == node_b->get_vector<T>();
            HANDLE_CASE(boolean, bool);
            HANDLE_CASE(i4, int8_t);
            HANDLE_CASE(u4, uint8_t);
            HANDLE_CASE(i32, int);
            HANDLE_CASE(i64, int64_t);
            HANDLE_CASE(f16, uint16_t);
            HANDLE_CASE(f32, float);
#undef HANDLE_CASE
        default:
            OPENVINO_THROW("Unable to handle type ", node_a->output(0));
        }
        return false;
    };
    auto check_and_mark = [&](const ov::npuw::RepeatedBlock::MatchedLayers& bank) {
        std::unordered_set<CTPtr> instances;
        for (auto&& l : bank) {
            instances.insert(const_cache.at(l));
        }
        auto& proto_node = *instances.begin();
        auto& proto_shape = proto_node->output(0).get_shape();

        LOG_DEBUG("Checking a bank with prototype node " << proto_node << "...");
        LOG_BLOCK();

        if ((((proto_shape.size() == 0 || (proto_shape.size() == 1 && proto_shape[0] <= 10)) &&
              proto_node->output(0).get_element_type().is_integral()) ||
             ((proto_node->output(0).get_element_type() == ov::element::f32 ||
               proto_node->output(0).get_element_type() == ov::element::f16) &&
              std::accumulate(proto_shape.begin(), proto_shape.end(), size_t{1}, std::multiplies<std::size_t>()) ==
                  1)) &&
            std::all_of(instances.begin(), instances.end(), [&](const CTPtr& other_node) -> bool {
                return (other_node->output(0).get_shape() == proto_node->output(0).get_shape()) &&
                       values_are_the_same(proto_node, other_node);
            })) {
            // Check passed for this group.
            LOG_DEBUG("[KEEP] It is safe to keep this bank in function");
            for (auto&& const_node : instances) {
                func_group.consts_to_keep.insert(const_node);
            }
        } else {
            LOG_DEBUG("[CUT ] This group of Const ops will be cut-off from the function");
        }
    };
    for (auto&& bank : rep_block.consts) {
        check_and_mark(bank);
    }
    for (auto&& bank : rep_block.scalars) {
        if (bank.size() > 1) {
            check_and_mark(bank);
        } else {
            // A single Constant layer in the match bank means _all_ the repeated
            // blocks refer to this constant (like in 1:N connection), so it is
            // ok to keep this const in the function body.
            NPUW_ASSERT(bank.size() == 1);
            func_group.consts_to_keep.insert(const_cache.at(*bank.begin()));
        }
    }
}

void Partitioner::matchParameters(const std::string& func_name) {
    LOG_VERB("Matching parameters for function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // In our repeated patterns, a Parameter is uniquely identified
    // by a set of its readers. A reader is a node+port.
    using PReader = std::pair<std::string, std::size_t>;
    using PKey = std::set<PReader>;

    auto& func = all_functions.at(func_name);
    auto& model_group = func.mdls;

    // FIXME: Here we rely on sacred knowledge that the first
    // subgraph in the list is the prototype... Use this to
    // initialize the proto parameter reader map
    std::map<PKey, PPtr> proto_parameters;
    {
        LOG_DEBUG("Generating proto pkeys...");
        LOG_BLOCK();
        auto& body = model_group.front();
        std::unordered_set<ov::Node*> this_model_nodes;
        for (auto&& node_ptr : body->get_ordered_ops()) {
            this_model_nodes.insert(node_ptr.get());
        }
        for (auto&& node : body->get_ordered_ops()) {
            if (ov::op::util::is_parameter(node)) {
                PKey pkey;
                for (auto&& iport : node->output(0).get_target_inputs()) {
                    if (this_model_nodes.count(iport.get_node()) > 0) {
                        LOG_DEBUG("Register link " << iport.get_node()->get_friendly_name() << " : "
                                                   << iport.get_index());
                        pkey.insert(PReader{iport.get_node()->get_friendly_name(), iport.get_index()});
                    }
                }
                LOG_DEBUG("Stored " << node);
                proto_parameters[pkey] = std::dynamic_pointer_cast<PPtr::element_type>(node);
            }
        }
    }

    // Now walk other submodels and match parameters with the same key
    // (yes, including the first one)
    for (std::size_t call_id = 0; call_id < model_group.size(); ++call_id) {
        LOG_DEBUG("Handle function call...");
        LOG_BLOCK();
        auto call = model_group[call_id];
        auto subg_ref = func.refs[call_id];

        std::unordered_set<ov::Node*> this_model_nodes;
        for (auto&& node_ptr : call->get_ordered_ops()) {
            this_model_nodes.insert(node_ptr.get());
        }
        for (auto&& node : call->get_ordered_ops()) {
            using ov::npuw::util::at::_;

            if (ov::op::util::is_parameter(node)) {
                PKey pkey;
                for (auto&& iport : node->output(0).get_target_inputs()) {
                    if (this_model_nodes.count(iport.get_node()) > 0) {
                        LOG_DEBUG("Register link " << iport.get_node()->get_friendly_name() << " : "
                                                   << iport.get_index());
                        pkey.insert(PReader{_(layer_to_prototype).at(iport.get_node()->get_friendly_name()),
                                            iport.get_index()});
                    }
                }
                LOG_DEBUG("Find orig parameter for " << node);
                auto& orig_param = proto_parameters.at(pkey);
                auto this_param = std::dynamic_pointer_cast<PPtr::element_type>(node);
                func.param_call_to_proto[SubgParam(subg_ref, this_param)] = orig_param;
            }
        }
    }
    LOG_VERB("Done");
}

void Partitioner::matchResults(const std::string& func_name) {
    LOG_VERB("Matching results for function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // FIXME: Here we rely on sacred knowledge that the first
    // subgraph in the list is the prototype...

    // In our repeated patterns, a Result is uniquely identified
    // by its writer. A writer is a node+port.
    using RKey = std::pair<std::string, std::size_t>;

    auto& func = all_functions.at(func_name);
    auto& model_group = func.mdls;

    // FIXME: Here we rely on sacred knowledge that the first
    // subgraph in the list is the prototype... Use this to
    // initialize the proto result writer map
    std::map<RKey, RPtr> proto_results;
    {
        auto& body = model_group.front();
        for (auto&& node : body->get_ordered_ops()) {
            if (ov::op::util::is_output(node)) {
                auto&& port = node->input(0).get_source_output();
                RKey rkey = {port.get_node()->get_friendly_name(), port.get_index()};
                proto_results[rkey] = std::dynamic_pointer_cast<RPtr::element_type>(node);
            }
        }
    }

    // Now walk all submodels and match parameters with the same key
    // (yes, including the first one)
    for (std::size_t call_idx = 0; call_idx < model_group.size(); ++call_idx) {
        auto call = model_group[call_idx];
        auto subg_ref = func.refs[call_idx];
        for (auto&& node : call->get_ordered_ops()) {
            if (ov::op::util::is_output(node)) {
                auto&& port = node->input(0).get_source_output();
                RKey rkey = {layer_to_prototype.at(port.get_node()->get_friendly_name()), port.get_index()};
                auto& orig_result = proto_results.at(rkey);
                auto this_result = std::dynamic_pointer_cast<RPtr::element_type>(node);
                func.result_call_to_proto[SubgResult(subg_ref, this_result)] = orig_result;
            }
        }
    }
    LOG_VERB("Done");
}

void Partitioner::createFunction(FunctionPipeline& func_ggg) {
    using namespace ov::npuw::weights;

    ov::npuw::Subgraph& body_sg = func_ggg.refs.front();
    const std::string func_name = body_sg._repeated_id;

    LOG_VERB("Registering a new function " << func_name << "...");
    LOG_BLOCK();

    ov::npuw::Subgraph funcall{};
    std::swap(body_sg._funcall, body_sg._repeated_id);  // FIXME: unclear

    // Preserve information about funcall and PARAMETERS for matching in
    // compiled_model (see m_inputs/outputs_to_submodels_inputs/outputs mess)
    funcall._funcall = body_sg._funcall;
    funcall._parameters = body_sg._parameters;
    funcall._results = body_sg._results;
    funcall._gflops = body_sg._gflops;  // preserving this is required for proper stats
    funcall._ops = body_sg._ops;        // preserving this is requried for proper stats
    funcall._avoid_list = body_sg._avoid_list;
    funcall._forced_to_fcall = body_sg._forced_to_fcall;

    // Declare a new function AND record a function call
    ov::npuw::Function function;
    function._model = func_ggg.mdls.front();
    function._param_offset = body_sg._parameters.size();
    function._tag = body_sg._tag;
    std::size_t new_param_idx = function._param_offset;

    for (auto&& node_ptr : function._model->get_ordered_ops()) {
        if (ov::op::util::is_parameter(node_ptr) || ov::op::util::is_constant(node_ptr) ||
            ov::op::util::is_output(node_ptr)) {
            // Skip Parameter, Const, and Result layers from bank matching
            continue;
        }
        const auto& this_layer_name = node_ptr->get_friendly_name();
        LOG_DEBUG("Processing " << this_layer_name);

        for (auto&& input_desc : node_ptr->inputs()) {
            const auto& prod_output = input_desc.get_source_output();
            const auto input_node = prod_output.get_node_shared_ptr();
            const auto iport = std::make_pair(node_ptr->get_friendly_name(), input_desc.get_index());
            LOG_DEBUG("Processing input " << prod_output);
            LOG_BLOCK();

            if (ov::op::util::is_constant(input_node) && func_ggg.consts_to_keep.count(input_node) == 0) {  // (n)/1/i
                LOG_DEBUG("Handling a Constant input " << prod_output);
                LOG_BLOCK();

                auto new_param = std::make_shared<ov::op::v0::Parameter>(prod_output.get_element_type(),
                                                                         prod_output.get_partial_shape());
                input_desc.replace_source_output(new_param);  // (n)/1/i/a
                function._model->add_parameters({std::move(new_param)});
                LOG_DEBUG("Register Parameter[" << new_param_idx << "] as input to " << iport.first << " / "
                                                << iport.second);
                function._param_mapping[iport] = new_param_idx;  // (n)/1/i/b
                new_param_idx++;

                LOG_DEBUG("Register " << prod_output << " in the function closure");
                funcall._lazy_closure.push_back(
                    LazyTensor(TransformType::THIS,
                               std::static_pointer_cast<ov::op::v0::Constant>(input_node)));  // (n)/1/i/c
            } else if (ov::op::util::is_parameter(input_node)) {
                LOG_DEBUG("Handling a Parameter input " << prod_output);
                LOG_BLOCK();

                // Record that this node also reads a parameter.
                // Parameter must be already defined (originally)
                auto param_iter = std::find(body_sg._parameters.begin(),
                                            body_sg._parameters.end(),
                                            std::static_pointer_cast<ov::op::v0::Parameter>(input_node));
                if (param_iter == body_sg._parameters.end()) {
                    OPENVINO_THROW("NPUW:", input_node, " is not found in the subgraph's parameters");
                }
                const auto existing_param_idx = std::distance(body_sg._parameters.begin(), param_iter);
                LOG_DEBUG("Register Parameter[" << existing_param_idx << "] as input to " << iport.first << " / "
                                                << iport.second);
                function._param_mapping[iport] = existing_param_idx;
            }  // if(Const|Parameter)
        }      // for(inputs)
    }          // for(nodes)
    funcall._closure.resize(funcall._lazy_closure.size());
    function._num_params_total = new_param_idx;
    function._model->validate_nodes_and_infer_types();
    P.functions.insert({func_name, std::move(function)});

    // Write down the funcall to the list of subgraphs
    std::swap(funcall, body_sg);
    LOG_VERB("Done: " << func_name);
}

void Partitioner::identifySpatialRange(ov::npuw::Function& f) {
    NPUW_ASSERT(f._tag == "compute");

    // NB: The current logic must be changed. Here we assume we only
    // apply this change to "compute" subgraphs which we identify
    // based on well-known patterns. This won't work in the generic case.

    // The current logic is the following:
    // - Assume the function results are ALL SPATIAL (and this alone
    //   is a very strong assumption)
    // - Identify their SPATIAL dimension (which is dim[1] because
    //   we know how COMPUTE subgraphs are organized)
    // - Walk over the parameters (up to _param_offset), find
    //   spatial Parameters based on the dim we're looking at
    // - Report the findings.
    // Hence, the logic is not robust enough and should be generalized
    // in the future.

    // First, check our assumption on the function results
    const auto& f_results = f._model->get_results();
    NPUW_ASSERT(f_results.size() > 0);

    const auto& f_result_0 = f_results.front();
    const auto& f_result_0_shape = f_result_0->get_shape();

    if (f_result_0_shape.size() != 3) {
        return;  // NB: this is the only case we enable now
    }

    if (f_result_0_shape[1] <= 1) {
        return;  // NB: this is the only spatial dim we enable now
    }

    for (auto&& f_result_i : f_results) {
        // Yes, it will also compare r[0] vs r[0]
        const auto& f_result_i_shape = f_result_i->get_shape();
        if (f_result_0_shape.size() != f_result_i_shape.size()) {
            return;  // Do nothing
        }

        if (f_result_0_shape[1] != f_result_i_shape[1]) {
            return;  // Do nothing
        }
    }

    // Now, find the parameters with the same spatial dim
    // NB: again, this is a very weak feature to look for
    const auto& f_params = f._model->get_parameters();
    NPUW_ASSERT(f_params.size() > 0);

    using S = ov::npuw::function::Spatial;
    S spatial;
    spatial._range = f_result_0_shape[1];
    spatial._out_dim = 1;  // the only case we're looking into now

    for (std::size_t i = 0u; i < f._param_offset; i++) {
        const auto& f_param = f_params[i];
        const auto& f_param_dims = f_param->get_shape();

        auto spatial_dim_iter = std::find(f_param_dims.begin(), f_param_dims.end(), spatial._range);
        if (spatial_dim_iter != f_param_dims.end()) {
            std::size_t spatial_dim_idx = std::distance(f_param_dims.begin(), spatial_dim_iter);
            spatial._inputs.push_back(S::Param{f_param, spatial_dim_idx});
        }
    }

    // Apply the spatial change
    f._spatial = std::move(spatial);
}

void Partitioner::createFunction(const std::string& func_name) {
    createFunction(all_functions.at(func_name));
}

void Partitioner::matchRepeatedSubgraphs(const std::string& func_name) {
    using namespace ov::npuw::weights;

    LOG_VERB("Process function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    auto& func_ggg = all_functions.at(func_name);
    auto& model_group = func_ggg.mdls;
    auto& func_group = func_ggg.refs;

    // First create a function body - take the first subgraph from
    // the funcall group as a prototype. This operation is identical
    // to converting a subgraph to a CWAI form
    createFunction(func_ggg);

    auto& body_params = func_group.front().get()._parameters;
    auto& body_results = func_group.front().get()._results;

    // Now process all other subgraphs as function calls
    auto mod_iter = model_group.begin() + 1;  // FIXME: use zip
    for (auto iter = func_group.begin() + 1; iter != func_group.end(); ++iter, ++mod_iter) {
        ov::npuw::Subgraph& this_sg = *iter;
        std::swap(this_sg._funcall, this_sg._repeated_id);

        ov::npuw::Subgraph funcall{};
        funcall._funcall = func_name;
        funcall._parameters = this_sg._parameters;  // duplicated code again!
        funcall._results = this_sg._results;        // duplicated code again!
        funcall._gflops = this_sg._gflops;          // duplicated code again!
        funcall._ops = this_sg._ops;                // duplicated code again!
        funcall._avoid_list = this_sg._avoid_list;  // duplicated code again!
        funcall._forced_to_fcall = this_sg._forced_to_fcall;
        rearrange_to_function_protocol(this_sg, body_params, funcall._parameters, func_ggg.param_call_to_proto, true);
        rearrange_to_function_protocol(this_sg, body_results, funcall._results, func_ggg.result_call_to_proto);

        auto func_iter = P.functions.find(func_name);
        NPUW_ASSERT(func_iter != P.functions.end());

        // Just record a function call of the existing function
        LOG_DEBUG("Registering a call to function " << func_name << "...");
        LOG_BLOCK();
        const auto& function = func_iter->second;
        funcall._closure.resize(function._num_params_total - function._param_offset);
        funcall._lazy_closure.resize(function._num_params_total - function._param_offset);

        auto tmp_model = *mod_iter;
        for (auto&& node_ptr : tmp_model->get_ordered_ops()) {
            if (ov::op::util::is_parameter(node_ptr) || ov::op::util::is_constant(node_ptr) ||
                ov::op::util::is_output(node_ptr)) {
                // Skip Parameter, Const, and Result layers from bank matching
                continue;
            }

            const auto& this_layer_name = node_ptr->get_friendly_name();
            const auto& proto_layer_name = layer_to_prototype.at(this_layer_name);  // (t)/1/a

            for (auto&& input_desc : node_ptr->inputs()) {
                const auto& prod_output = input_desc.get_source_output();
                const auto input_node = prod_output.get_node_shared_ptr();
                if (ov::op::util::is_constant(input_node) && func_ggg.consts_to_keep.count(input_node) == 0) {
                    auto param_idx = function._param_mapping.at(
                        std::make_pair(proto_layer_name, input_desc.get_index()));  // (t)/1/b
                    LOG_DEBUG("Register " << prod_output << " in the function closure[" << param_idx
                                          << "] (via prototype " << proto_layer_name << ")");
                    funcall._lazy_closure[param_idx - function._param_offset] =
                        LazyTensor(TransformType::THIS,
                                   std::static_pointer_cast<ov::op::v0::Constant>(input_node));  // (t)/1/c
                }
            }  // for (inputs)
        }      // for(nodes)

        // Write down the funcall to the list of subgraphs
        std::swap(funcall, this_sg);
        LOG_DEBUG("Done: funcall(" << func_name << ")");
    }  // for(rest of models)

    LOG_VERB("Done");
}

void Partitioner::spatial(const std::string& func_name) {
    ov::npuw::Function& f = P.functions.at(func_name);

    // Identify the spatial dimension for this function
    // Works only for Compute case.
    // FIXME: Replace this string identification with smt better
    if (!cfg.get<::intel_npu::NPUW_SPATIAL>() || f._tag != "compute") {
        LOG_VERB("No spatial optimizations will be done to  " << func_name << " in model " << model->get_friendly_name()
                                                              << "...");
        return;
    }

    LOG_VERB("Turn " << func_name << " into spatial execution in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    identifySpatialRange(f);
    if (!f._spatial) {
        LOG_WARN("No spatial ranges identified in the COMPUTE block, expect a higher compile time");
        return;
    }

    LOG_VERB("Spatial range: " << f._spatial->_range);

    // Final check before transformations
    f._spatial->_slice = cfg.get<::intel_npu::NPUW_SPATIAL_NWAY>();
    if (f._spatial->_slice == 0) {
        LOG_WARN("NWAY is set to 0, disabling it (but better disable SPATIAL setting itself)");
        f._spatial.reset();  // Erase spatial information to avoid conflicts
        return;
    }

    // Apply transformation to the model. Note: only function body is modified
    // Accumulate the reshape map
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
    for (auto&& p : f._spatial->_inputs) {
        ov::Shape shape = p.param->get_shape();
        shape[p.dim] = f._spatial->_slice;
        new_shapes[p.param->output(0)] = shape;
    }
    f._model->reshape(new_shapes);

    LOG_VERB("Done");
}

void Partitioner::optimize(const std::string& func_name) {
    using namespace ov::npuw::weights;

    ov::npuw::Function& f = P.functions.at(func_name);
    auto& func_group = all_functions.at(func_name);

    auto do_permute = [&](ov::npuw::patterns::opt::Context& ctx) {
        for (auto&& p : ctx.closures_to_permute) {
            auto param_idx = f._model->get_parameter_index(p.first);
            auto closure_idx = param_idx - f._param_offset;
            ov::parallel_for(func_group.refs.size(), [&](std::size_t f_idx) {
                auto& funcall = func_group.refs[f_idx].get();
                funcall._lazy_closure[closure_idx].update(TransformType::PERMUTE, p.second);
            });
        }
    };
    auto do_cvtf16 = [&](ov::npuw::patterns::opt::Context& ctx) {
        for (auto&& p : ctx.closures_to_f16) {
            auto param_idx = f._model->get_parameter_index(p);
            auto closure_idx = param_idx - f._param_offset;
            ov::parallel_for(func_group.refs.size(), [&](std::size_t f_idx) {
                auto& funcall = func_group.refs[f_idx].get();
                funcall._lazy_closure[closure_idx].update(TransformType::CONVERT, std::monostate{});
            });
        }
    };

    // Regardless of DQ setting, run this first
    {
        ov::npuw::patterns::opt::Context ctx;
        ctx.is_spatial = f._spatial.has_value();
        ctx.pmm_dims = cfg.get<::intel_npu::NPUW_PMM>();

        // Run Head/Tail passes
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::opt::DQUnpackDictGatheru>(std::ref(ctx));
        rewr.add_matcher<ov::npuw::patterns::opt::DQUnpackDictGatherGQi>(std::ref(ctx));
        rewr.add_matcher<ov::npuw::patterns::opt::DQUnpackDictMatMulCWu>(std::ref(ctx));
        // NB: This pass is disabled for reason! It doesn't make things better
        // rewr.add_matcher<ov::npuw::patterns::opt::DQUnpackDictMatMulGQi>(std::ref(ctx));
        rewr.add_matcher<ov::npuw::patterns::opt::CompressDictMatMulf32>(std::ref(ctx));
        rewr.add_matcher<ov::npuw::patterns::opt::DQParMMGQ>(std::ref(ctx));
        rewr.run_on_model(f._model);

        // Move Gather to host, if required
        if (cfg.get<::intel_npu::NPUW_HOST_GATHER>()) {
            ov::pass::GraphRewrite rewr2;
            rewr2.add_matcher<ov::npuw::patterns::opt::HostGather>(std::ref(ctx));
            rewr2.add_matcher<ov::npuw::patterns::opt::HostGatherDQ>(std::ref(ctx));
            rewr2.run_on_model(f._model);
        }

        // Run parallel matmul merge
        mergeParallelMatMuls(f._model, ctx);

        ov::ParameterVector new_params;
        std::vector<ov::npuw::patterns::opt::Context::PPtr> to_remove;
        std::set<std::size_t> to_remove_idx;

        // Concatenate closures for "concatenated" parameters
        for (auto&& p : ctx.params_to_concat) {
            new_params.push_back(p.first);
            const auto& params_to_concat = p.second.first;
            const auto axis = p.second.second;

            std::vector<std::size_t> to_concat_idx;
            for (auto&& p_to_concat : params_to_concat) {
                auto p_to_concat_idx = f._model->get_parameter_index(p_to_concat);
                to_remove.push_back(p_to_concat);
                to_concat_idx.push_back(p_to_concat_idx - f._param_offset);
                to_remove_idx.insert(p_to_concat_idx);
            }
            ov::parallel_for(func_group.refs.size(), [&](std::size_t f_idx) {
                auto& funcall = func_group.refs[f_idx].get();
                std::vector<LazyTensor> to_concat;
                // Fill tensor vector
                for (auto&& cidx : to_concat_idx) {
                    // FIXME: Assuming here concat goes first and other transformations later.
                    //        This allows to store ov::Tensor and ignore their potential history of transformations
                    NPUW_ASSERT(!funcall._lazy_closure[cidx].has_transformations());
                    to_concat.push_back(funcall._lazy_closure[cidx]);
                }
                // Note: we can ignore updating funcall._lazy_closure[cidx] here since those LazyTensors will be gone
                // and the new one added into the vector
                if (!to_concat.empty()) {
                    funcall._lazy_closure.push_back(LazyTensor(TransformType::CONCAT, std::make_pair(to_concat, axis)));
                    // Some of the tensors might be in closure - preserve it's 1:1 idx mapping with _lazy_closure
                    funcall._closure.push_back(ov::Tensor());
                }
            });
        }

        // Unpack closures in compile time, where requested
        for (auto&& p : ctx.params_to_unpack) {
            const auto& tensor_to_unpack = p.second;
            auto w_idx = f._model->get_parameter_index(tensor_to_unpack.w);
            auto z_idx = f._model->get_parameter_index(tensor_to_unpack.z);
            auto s_idx = f._model->get_parameter_index(tensor_to_unpack.s);

            new_params.push_back(p.first);
            to_remove.push_back(tensor_to_unpack.w);
            to_remove.push_back(tensor_to_unpack.s);
            to_remove_idx.insert(w_idx);
            to_remove_idx.insert(s_idx);

            if (tensor_to_unpack.z) {
                to_remove.push_back(tensor_to_unpack.z);
                to_remove_idx.insert(z_idx);
            }

            ov::parallel_for(func_group.refs.size(), [&](std::size_t f_idx) {
                auto& funcall = func_group.refs[f_idx].get();
                // FIXME: assuming no transformations were applied to the tensor - since we are utilizing the original
                // ov::Tensor below
                LazyTensor cw = funcall._lazy_closure[w_idx - f._param_offset];
                LazyTensor cz = z_idx != -1 ? funcall._lazy_closure[z_idx - f._param_offset]
                                            : LazyTensor(TransformType::THIS, ov::Tensor());
                LazyTensor cs = funcall._lazy_closure[s_idx - f._param_offset];

                // FIXME: currently there is an issue that we don't share such tensor between head and tail
                funcall._lazy_closure.push_back(
                    LazyTensor(TransformType::UNPACK,
                               std::make_tuple(cw, cz, cs, p.first->get_shape(), p.first->get_element_type())));
                // Some of the tensors might be in closure - preserve it's 1:1 idx mapping with _lazy_closure
                funcall._closure.push_back(ov::Tensor());
            });
        }

        // Convert parameters to f16 where required
        do_cvtf16(ctx);

        // Host-side gather, pt 1. Add new parameters first
        if (ctx.params_to_gather) {
            auto& params_to_gather = *ctx.params_to_gather;
            new_params.push_back(params_to_gather.pnew);
            for (auto&& funcall : func_group.refs) {
                auto new_elem_type = params_to_gather.pnew->get_element_type();
                const auto& new_shape = params_to_gather.pnew->get_shape();
                // Note: no allocation needed for this tensor - set to _closure and dummy in _lazy_closure
                // FIXME: It turns out this tensor will be completely unused.
                // It will just sit in the memory to do nothing.
                // Most likely it may stay empty since we need a 1:1 matching between
                // closure tensors and parameters (minus base).
                // Based on our logic (when tensors get transferred from lazy tensors via bank
                // to the closure), this tensor should be non-empty to avoid this process.
                funcall.get()._closure.push_back(ov::Tensor(new_elem_type, new_shape));
                funcall.get()._lazy_closure.push_back(LazyTensor(TransformType::THIS, ov::Tensor()));
            }
        }

        // Add all new parameters introduced by this change
        f._model->add_parameters(new_params);

        // Remove LazyTensors which will be concatenated
        for (auto&& fref : func_group.refs) {
            auto& funcall = fref.get();
            std::vector<LazyTensor> new_transforms;
            // Some of the tensors might be in closure (e.g. host-gather), thus need to preserve the _closure as well
            std::vector<ov::Tensor> new_closure;
            for (std::size_t tidx = 0; tidx < funcall._lazy_closure.size(); tidx++) {
                if (to_remove_idx.count(f._param_offset + tidx) == 0) {
                    new_transforms.push_back(funcall._lazy_closure[tidx]);
                    new_closure.push_back(funcall._closure[tidx]);
                }
            }
            funcall._lazy_closure = std::move(new_transforms);
            funcall._closure = std::move(new_closure);
        }
        // Remove parameters that were concatenated
        for (auto&& now_remove : to_remove) {
            f._model->remove_parameter(now_remove);
        }

        f._model->validate_nodes_and_infer_types();

        // Host-side gather, pt. 2: Write the gather mappings to funcall
        if (ctx.params_to_gather) {
            auto& params_to_gather = *ctx.params_to_gather;
            auto gather_dst_id = f._model->get_parameter_index(params_to_gather.pnew);
            auto gather_src_id = f._model->get_parameter_index(params_to_gather.pold);
            auto gather_idx_id = f._model->get_parameter_index(params_to_gather.pids);
            for (auto&& funcall : func_group.refs) {
                funcall.get()._host_gather = ov::npuw::Subgraph::Gather{gather_dst_id, gather_src_id, gather_idx_id};
            }
        }
    }

    if (!cfg.get<::intel_npu::NPUW_DQ>()) {
        LOG_VERB("No optimizations will be done to  " << func_name << " in model " << model->get_friendly_name()
                                                      << "...");
        return;
    }

    LOG_VERB("Optimize function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // Run "dynamic quantization"
    ov::npuw::patterns::opt::Context ctx;
    ctx.is_spatial = f._spatial.has_value();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulCWi>();
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQi>(std::ref(ctx));
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQ2i>(std::ref(ctx));
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQiP>(std::ref(ctx));
    rewr.add_matcher<ov::npuw::patterns::opt::DQMatMulGQ2iP>(std::ref(ctx));
    rewr.run_on_model(f._model);
    ov::pass::Validate().run_on_model(f._model);

    do_permute(ctx);
    do_cvtf16(ctx);

    LOG_VERB("Done");
}

void Partitioner::decompressionCutOff(const std::string& func_name) {
    LOG_VERB("Decompression cut-off for function " << func_name << " in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    std::string dcoff_type_opt = cfg.getString<::intel_npu::NPUW_DCOFF_TYPE>();
    ov::element::Type dcoff_type{};
    if (!dcoff_type_opt.empty()) {
        if (dcoff_type_opt == "i8") {
            dcoff_type = ov::element::i8;
        } else if (dcoff_type_opt == "f16") {
            dcoff_type = ov::element::f16;
        } else if (dcoff_type_opt == "f32") {
            dcoff_type = ov::element::f32;
        } else {
            OPENVINO_THROW("Unknwon dcoff type: ", dcoff_type_opt);
        }
    } else {
        LOG_VERB("Cancelled - no dcoff type specified via " << ::intel_npu::NPUW_DCOFF_TYPE().key() << " property.");
        return;
    }

    ov::npuw::DCOffMode dcoff_mode = ov::npuw::DCOffMode::CAST_ONLY;
    if (cfg.get<::intel_npu::NPUW_DCOFF_SCALE>()) {
        if (dcoff_type != ov::element::f16) {
            // Moving out scaling off graph works only for the f16 DCOFF type
            LOG_WARN(::intel_npu::NPUW_DCOFF_SCALE().key()
                     << " property is specified, but the target " << ::intel_npu::NPUW_DCOFF_TYPE().key()
                     << " is not f16 - ignoring");
        } else {
            LOG_VERB("Decompression cut-off: Weight scaling will be moved off the model");
            dcoff_mode = ov::npuw::DCOffMode::CAST_SCALE;
        }
    }

    // What we do here? Identify the decompression patterns (see
    // patterns/dcoff.hpp) in our function, put an extra Convert and
    // remember which Constants to decompress in the function
    // prologue, during the execution.

    ov::npuw::Function& f = P.functions.at(func_name);
    LOG_DEBUG("Function model inputs before the DCOFF:");
    for (auto&& input : f._model->inputs()) {
        LOG_BLOCK();
        LOG_DEBUG(input);
    }
    LOG_DEBUG("Running the graph transformations...");
    {
        LOG_BLOCK();

        ov::npuw::patterns::DCOFFParams params_to;

        ov::pass::GraphRewrite rewr;
        // Old LLaMa-v2 patterns (Symmetric)
        rewr.add_matcher<ov::npuw::patterns::SymmNoZP::DCOFFPassMatMul>(dcoff_mode, dcoff_type, std::ref(params_to))
            ->build();
        rewr.add_matcher<ov::npuw::patterns::SymmNoZP::DCOFFPassGather>(dcoff_mode, dcoff_type, std::ref(params_to))
            ->build();

        // ChatGLM (GPTQ) and New LLaMa-v2 patterns (Symmetric)
        rewr.add_matcher<ov::npuw::patterns::SymmZP::DCOFFPassReshape1>(dcoff_mode, dcoff_type, std::ref(params_to))
            ->build();
        rewr.add_matcher<ov::npuw::patterns::SymmZP::DCOFFPassConvert1>(dcoff_mode, dcoff_type, std::ref(params_to))
            ->build();

        // LLaMaGPTQ
        rewr.add_matcher<ov::npuw::patterns::SymmZP::DCOFFPassReshape2>(dcoff_mode, dcoff_type, std::ref(params_to));

        // Phi-3 4SymW16A
        rewr.add_matcher<ov::npuw::patterns::SymmZP::DCOFFPassReshape3>(dcoff_mode, dcoff_type, std::ref(params_to));

        // Phi-3 i4 4SymW16A
        rewr.add_matcher<ov::npuw::patterns::SymmZP::DCOFFPassReshape4>(dcoff_mode, dcoff_type, std::ref(params_to));

        // Asymmetric zeropoints
        rewr.add_matcher<ov::npuw::patterns::AsymmZP::DCOFFPassReshape>(dcoff_mode, dcoff_type, std::ref(params_to));

        rewr.run_on_model(f._model);

        ov::pass::Validate val;
        val.run_on_model(f._model);

        if (!params_to.scales.empty()) {
            LOG_VERB("Weight scaling was removed from the function body"
                     " -- updating the function closure");
            LOG_BLOCK();

            // Build a closure remap here: which input tensors are no
            // longer parameters but move to the unpack procedure?
            auto closure_remap = ov::npuw::patterns::build_remap(f, params_to);

            // Now modify the each function call's closure accordingly
            // according to the remaps.
            auto& func_group = all_functions.at(func_name);
            for (auto&& fref : func_group.refs) {
                auto& funcall = fref.get();
                ov::npuw::patterns::apply_remap(funcall, closure_remap);
            }

            // Finally, remove the function body's parameters here
            ov::npuw::patterns::finalize_remap(f, func_group.refs.front(), closure_remap);
        }  // if (CAST_SCALE && have(params_to_scale))
    }
    LOG_DEBUG("Function model inputs after the DCOFF:");
    for (auto&& input : f._model->inputs()) {
        LOG_BLOCK();
        LOG_DEBUG(input);
    }
    LOG_VERB("Done");
}

void Partitioner::finalizeLinks() {
    LOG_VERB("Finalizing links in model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    // Write down the [subgraph_i][out_j] -> [subgraph_k][input_n]
    // mapping here. I/J & K/N indices become final at this point

    // FIXME: make it a template helper?
    auto get_idx_param = [this](std::size_t subgr_idx_to, const PPtr& ptr) -> std::size_t {
        auto& sg_desc = P.subgraphs[subgr_idx_to];
        if (sg_desc._funcall.empty()) {
            // Not a function call (or a sole function call):
            // find in a subgraph itself
            auto& params = sg_desc._parameters;
            auto param_iter = std::find(params.begin(), params.end(), ptr);
            NPUW_ASSERT(param_iter != params.end());
            return std::distance(params.begin(), param_iter);
        } else {
            // A function call: find in the prototype subgraph
            auto& params = P.functions.at(sg_desc._funcall)._model->get_parameters();
            auto& proto = func_pipeline_type == FunctionPipelineType::CWAI
                              ? ptr  // no protos in the CWAI case..
                              : all_functions.at(sg_desc._funcall).param_call_to_proto.at(SubgParam(sg_desc, ptr));
            auto param_iter = std::find(params.begin(), params.end(), proto);
            NPUW_ASSERT(param_iter != params.end());
            return std::distance(params.begin(), param_iter);
        }
    };

    auto get_idx_result = [this](std::size_t subgr_idx_from, const RPtr& ptr) -> std::size_t {
        auto& sg_desc = P.subgraphs[subgr_idx_from];
        if (sg_desc._funcall.empty()) {
            // Not a function call (or a sole function call):
            // find in a subgraph itself
            auto& results = sg_desc._results;
            auto result_iter = std::find(results.begin(), results.end(), ptr);
            NPUW_ASSERT(result_iter != results.end());
            return std::distance(results.begin(), result_iter);
        } else {
            // A function call: find in the prototype subgraph
            auto& results = P.functions.at(sg_desc._funcall)._model->get_results();
            auto& proto = func_pipeline_type == FunctionPipelineType::CWAI
                              ? ptr  // no protos in the CWAI case...
                              : all_functions.at(sg_desc._funcall).result_call_to_proto.at(SubgResult(sg_desc, ptr));
            auto result_iter = std::find(results.begin(), results.end(), proto);
            NPUW_ASSERT(result_iter != results.end());
            return std::distance(results.begin(), result_iter);
        }
    };

    ov::npuw::Links& subgraph_links = P.input_to_prev_output;
    for (auto&& ptr_link : subgraph_ptr_links) {
        // what happens when either sg_to or sg_from is a function
        // call? These vectors are basically empty in this case...
        // Needs to be matched with the function body parameter/
        // result order.. but how? But how... see above, the complexity
        // is there.

        std::size_t subgraph_idx_to;
        PPtr subgraph_param_to;
        std::tie(subgraph_idx_to, subgraph_param_to) = ptr_link.first;
        auto param_idx = get_idx_param(subgraph_idx_to, subgraph_param_to);

        std::size_t subgraph_idx_from;
        RPtr subgraph_result_from;
        std::tie(subgraph_idx_from, subgraph_result_from) = ptr_link.second;
        auto result_idx = get_idx_result(subgraph_idx_from, subgraph_result_from);

        subgraph_links[ov::npuw::LinkTo{subgraph_idx_to, param_idx}] =
            ov::npuw::LinkFrom{subgraph_idx_from, result_idx};

        LOG_BLOCK();
        LOG_DEBUG("Record link [" << subgraph_idx_to << "]:" << param_idx << "  <---  [" << subgraph_idx_from << "]/"
                                  << result_idx);
    }
    LOG_VERB("Done");
}

}  // namespace

ov::npuw::Partitioning ov::npuw::getPartitioning(const std::shared_ptr<ov::Model>& model, ::intel_npu::Config& cfg) {
    LOG_INFO("Building partitioning for model " << model->get_friendly_name() << "...");
    LOG_BLOCK();

    ov::npuw::Ensemble ens;

    // Try to load the partitioning plan...
    const std::string file_path = cfg.get<::intel_npu::NPUW_PLAN>();
    if (file_path.empty()) {
        LOG_INFO("No " << ::intel_npu::NPUW_PLAN().key() << " property is provided! Using online partitioning.");
        ens = ov::npuw::online::buildPartitioning(model, cfg);
    } else {
        ens = load_groups(model, file_path);
    }

    const bool dump_full_opt = cfg.get<::intel_npu::NPUW_DUMP_FULL>();

    if (dump_full_opt) {
        ov::save_model(model, model->get_friendly_name() + ".xml");
        LOG_INFO("Dumped the model in the current directory.");
    }

    if (ens.groups.empty()) {
        LOG_INFO("Fall-back to single-model inference.");
        // So far we have 1 subgraph which is the whole network
        Subgraph subgraph;
        subgraph._parameters = model->get_parameters();
        subgraph._results = model->get_results();
        subgraph._sinks = model->get_sinks();
        return Partitioning{std::vector<Subgraph>{std::move(subgraph)}};
    }

    // Handle funcall everywhere, if needed
    FuncallEverywhere fcew(model, cfg);
    if (fcew.enabled()) {
        std::size_t gid = 0u;  // TODO: Use indexed()
        for (auto&& this_group : ens.groups) {
            if (!this_group.repeated_id.empty()) {
                fcew.register_known(this_group.repeated_id);
            } else {
                auto new_id = fcew.register_new();
                LOG_INFO("Turning block " << gid << " into a function " << this_group.repeated_id << "...");
                LOG_BLOCK();
                this_group.repeated_id = std::move(new_id);
                this_group.forced_to_fcall = true;

                ov::npuw::RepeatedBlock this_block;
                for (const auto& layer : this_group.all_layers) {
                    // Note: NOT move(layer)! It breaks the code here.
                    this_block.matches.push_back(ov::npuw::RepeatedBlock::MatchedLayers{layer});
                }
                ens.repeated[this_group.repeated_id] = std::move(this_block);
                LOG_INFO("Done.");
            }
            gid++;
        }  // for(ens.groups)
    }      // if(fcew.enabled)

    Partitioning P;
    P.total_gflops = ens.gflops;

    Partitioner p(model, ens, P, cfg);
    p.identifySubgraphs();

    if (!ens.repeated.empty()) {
        if (cfg.get<::intel_npu::NPUW_FOLD>()) {
            // Do full-featured folding
            auto all_functions = p.initFunctionPipeline(Partitioner::FunctionPipelineType::FOLD);
            for (auto&& func_group : all_functions) {
                LOG_INFO("FOLD: Process function " << func_group << "...");
                LOG_BLOCK();
                p.propagateSlices(func_group);
                p.propagateConverts(func_group);
                p.propagateWeights(func_group);
                p.propagateScalars(func_group);
                p.propagateConvertsOut(func_group);
                p.sanityCheck(func_group);
                p.saveRepeatedConstants(func_group);
                p.matchParameters(func_group);
                p.matchResults(func_group);
                p.matchRepeatedSubgraphs(func_group);
                p.spatial(func_group);
                p.optimize(func_group);
                p.decompressionCutOff(func_group);
            }
        } else if (cfg.get<::intel_npu::NPUW_CWAI>()) {
            // Less brutal version - just transform repeated blocks
            // into the closure forms, but don't do folding.
            // This path is likely to be removed soon (is here for
            // debug purposes only, but doesn't have much practical
            // sense).
            auto all_functions = p.initFunctionPipeline(Partitioner::FunctionPipelineType::CWAI);
            for (auto&& func_group : all_functions) {
                LOG_INFO("CWAI: Process function " << func_group << "...");
                LOG_BLOCK();
                p.saveTinyConstants(func_group);
                p.saveScaleFactors(func_group);
                p.createFunction(func_group);
                p.decompressionCutOff(func_group);
            }
        } else {
            LOG_INFO("Note: Repeated blocks are found in the model " << model->get_friendly_name()
                                                                     << ", but folding or eager mode are not enabled");
        }
    }
    p.finalizeLinks();
    return P;
}
