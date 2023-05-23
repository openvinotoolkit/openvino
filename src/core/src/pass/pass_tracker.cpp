#include <utility>
#include <fstream>

#include "openvino/pass/pass_tracker.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/util/env_util.hpp"


template <typename Container>
std::string join(const Container& c, const char* glue = ", ") {
    std::stringstream oss;
    const char* s = "";
    for (const auto& v : c) {
        oss << s << v;
        s = glue;
    }
    return oss.str();
}

class AttrsSerializer : public ngraph::AttributeVisitor {
    std::map<std::string, std::string>& m_attrs;

    template <typename T>
    std::string create_atribute_list(ngraph::ValueAccessor<std::vector<T>>& adapter) {
        return join(adapter.get());
    }

public:
    AttrsSerializer(std::map<std::string, std::string>& attrs) : m_attrs(attrs) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        // TODO: to copy-paste from XMLSerializer
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        m_attrs[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        m_attrs[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        m_attrs[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        m_attrs[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int>>& adapter) override {
        m_attrs[name] = create_atribute_list(adapter);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_attrs[name] = create_atribute_list(adapter);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_attrs[name] = create_atribute_list(adapter);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_attrs[name] = create_atribute_list(adapter);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_attrs[name] = create_atribute_list(adapter);
    }
};

// Lightweight Graph representation which is used to store intermediate sub-graph
// that we serialize to JSON for pass_viz.
struct Graph {
    struct Vertex {
        explicit Vertex(ov::Node * node)
                : id(node->get_instance_id())
                , operation_type(node->get_type_name())
                , operation_name(node->get_friendly_name()){
            const auto rt_info = node->get_rt_info();
            if (rt_info.count("depth")) {
                depth = rt_info.at("depth").as<size_t>();
            } else {
                depth = 0;
            }

            if (rt_info.count("vertex_type")) {
                vertex_type = rt_info.at("vertex_type").as<std::string>();
            }

            AttrsSerializer attrsSerializer(operation_attrs);
            node->visit_attributes(attrsSerializer);
        }
        size_t id;
        size_t depth;
        std::string operation_type;
        std::string vertex_type;
        std::string operation_name;
        std::map<std::string, std::string> operation_attrs;


        std::string get_name() const { return std::to_string(id) + ":" + operation_type; }

        bool operator< (const Vertex& other) const { return id < other.id; }

        bool operator== (const Vertex & other) const {
            return id == other.id &&
                   depth == other.depth &&
                   operation_type == other.operation_type;
        }
    };

    bool is_empty() const { return value.empty(); }

    std::map<Vertex, std::set<Vertex>> value;
};

// ------------------------------ Lightweight JSON serializer -------------------------------------------

class JSONBase {
public:
    std::string m_name;
    std::string m_depth;
    std::vector<std::shared_ptr<JSONBase>> m_elements;

public:
    JSONBase(std::string name) : m_name(std::move(name)) {}

    virtual void serialize(std::ostream& out) const {};

    friend std::ostream& operator<< (std::ostream& out, const JSONBase& element) {
        element.serialize(out);
        return out;
    }
};

template <class T>
void serialize_value(std::ostream& out, const T & values) {
    out << '[';
    size_t index = 0;
    for (const auto & value : values) {
        out << '\"' << value << '\"' << ((index + 1 < values.size()) ? ", " : "");
        ++index;
    }
    out << ']';
}

template <>
void serialize_value(std::ostream& out, const std::string & value) {
    out << '\"' << value << '\"';
}

template <class T>
class JSONElement : public JSONBase {
private:
    T m_value;
public:
    JSONElement(const std::string & name, T value = {})
        : JSONBase(name)
        , m_value(std::move(value)) {}

    void set_value(T value) {
        if (!m_elements.empty()) {
            throw std::runtime_error("JSONElement error: element should have either value or list of elements");
        }
        m_value = std::move(value);
    }

    template<class V>
    JSONElement<V>& add_element(std::string name, V value) {
        if (!m_value.empty()) {
            throw std::runtime_error("JSONElement error: element should have either value or list of elements");
        }

        auto el = std::make_shared<JSONElement<V>>(name, value);
        el->m_depth = m_depth + "    ";
        m_elements.push_back(el);
        return *el;
    }

    JSONElement<std::string>& add_element(std::string name) {
        return add_element<std::string>(name , "");
    }

    void serialize(std::ostream& out) const override {
        if (!m_name.empty()) {
            out << m_depth + '\"' + m_name + "\": ";
        }
        if (!m_value.empty()) {
            serialize_value(out, m_value);
        } else {
            bool is_empty = m_elements.empty();
            out << "{" << (is_empty ? "" : "\n");
            for (size_t i = 0; i < m_elements.size(); ++i) {
                out << *m_elements[i] << (i == m_elements.size() - 1 ? "" : ",") << "\n";
            }
            out << (is_empty ? "" : m_depth) << "}";
        }
    }

    friend std::ostream& operator<< (std::ostream& out, const JSONElement<T> & element) {
        element.serialize(out);
        return out;
    }
};

class JSONWriter {
private:
    std::string m_out_file_path;
    JSONElement<std::string> m_root;

    void serialize() {
        std::ofstream out(m_out_file_path);
        out << m_root;
        out.close();
    }
public:
    JSONWriter(std::string file_path)
        : m_out_file_path(std::move(file_path))
        , m_root("", "") {}

    JSONElement<std::string>& get_root() { return m_root; }

    ~JSONWriter() {
        serialize();
    }
};

// ------------------------------ Graph manipulation -------------------------------------------

void graph_to_json(const Graph & graph,
                   const std::string & name,
                   const size_t index,
                   const std::string & type,
                   const std::vector<std::pair<std::string, std::function<bool(const Graph::Vertex&)>>>& colors) {
    if (graph.is_empty()) return;

    static size_t graph_idx = 0;

    std::string run_name = ov::util::getenv_string("NGRAPH_PASS_TRACKING_RUN_NAME");
    std::string model_name = ov::util::getenv_string("NGRAPH_PASS_TRACKING_MODEL_NAME");
    std::string out_dir = ov::util::getenv_string("NGRAPH_PASS_TRACKING_OUTPUT_DIR", ".");

    const std::string out_file = out_dir + "/dump_" + run_name + "_" + model_name + "_" + std::to_string(graph_idx) + ".json";

    JSONWriter json(out_file);
    auto & root = json.get_root();
    root.add_element("run_name", run_name);
    root.add_element("model_name", model_name);
    root.add_element("pass_name", name);
    root.add_element("index", std::to_string(index));
    root.add_element("type", type);
    auto & edges = root.add_element("edges");

    // generate edges
    for (const auto& item : graph.value) {
        std::set<std::string> value;
        for (const auto & to : item.second) {
            // TODO: this is workaround to bypass edges where the destination node is not in the nodes list (rare case). Need to debug model_partitioning.
            if (graph.value.count(to)) {
                value.insert(to.get_name());
            }
        }
        if (!value.empty()) {
            edges.add_element(item.first.get_name(), value);
        }
    }

    auto & nodes = root.add_element("nodes");

    for (const auto & item : graph.value) {
        const auto & v = item.first;
        auto & node = nodes.add_element(v.get_name());

        auto & node_attrs = node.add_element("attrs");
        node_attrs.add_element("type", v.operation_type);
        node_attrs.add_element("name", v.operation_name);

        for (const auto & attr : v.operation_attrs) {
            node_attrs.add_element(attr.first, attr.second);
        }

        auto & dot_attrs = node.add_element("dot_attrs");
        for (const auto & color : colors) {
            if (color.second(v)) {
                dot_attrs.add_element("color", color.first);
            }
        }
    }

    graph_idx += 1;
}

Graph model_partitioning(const ov::NodeVector& nodes,
                         const std::function<bool(std::shared_ptr<ov::Node>)> & pred,
                         const std::string & vertex_type,
                         const size_t max_depth = 3) {
    auto is_used = [](std::shared_ptr<ov::Node> item) -> bool {
        return item->get_rt_info().count("used") != 0;
    };

    auto get_depth = [](std::shared_ptr<ov::Node> item) -> size_t {
        const auto & rt_info = item->get_rt_info();
        if (rt_info.count("depth")) {
            return rt_info.at("depth").as<size_t>();
        }
        return 0;
    };

    auto set_node_type = [=](std::shared_ptr<ov::Node> node) {
        size_t depth = get_depth(node);
        if (depth == 0) {
            node->get_rt_info()["vertex_type"] = vertex_type;
        }
    };

    struct state {
        state(std::shared_ptr<ov::Node> a, std::shared_ptr<ov::Node> b, bool f) : cur(a), prev(b), is_back_edge(f) {}
        std::shared_ptr<ov::Node> cur = nullptr;
        std::shared_ptr<ov::Node> prev = nullptr;
        bool is_back_edge = false;
    };

    auto traverse = [=](std::shared_ptr<ov::Node> node, Graph & graph) {
        std::deque<state> q{{node, nullptr, false}};
        while (!q.empty()) {
            auto cur = q.front().cur;
            auto prev = q.front().prev;
            auto is_back_edge = q.front().is_back_edge;

            cur->get_rt_info()["used"] = 1;
            set_node_type(cur);

            q.pop_front();

            const auto cur_depth = get_depth(cur);
            if (cur_depth >= max_depth) continue;

            if (!graph.value.count(Graph::Vertex(cur.get()))) {
                graph.value.insert({Graph::Vertex(cur.get()), {}});
            }

            if (prev) {
                if (is_back_edge) {
                    graph.value[Graph::Vertex(cur.get())].insert(Graph::Vertex(prev.get()));
                } else {
                    graph.value[Graph::Vertex(prev.get())].insert(Graph::Vertex(cur.get()));
                }
            }

            for (const auto & input : cur->input_values()) {
                auto input_node = input.get_node_shared_ptr();
                if (!is_used(input_node)) {
                    bool is_valid = pred(input_node);

                    input_node->get_rt_info()["depth"] = is_valid ? 0 : cur_depth + 1;
                    input_node->get_rt_info()["used"] = 1;

                    set_node_type(input_node);
                    q.push_front({input_node, cur, true});
                } else {
                    graph.value[Graph::Vertex(input_node.get())].insert(Graph::Vertex(cur.get()));
                }
            }
            for (const auto & output : cur->outputs()) {
                for (const auto &input : output.get_target_inputs()) {
                    auto input_node = input.get_node()->shared_from_this();
                    if (!is_used(input_node)) {
                        bool is_valid = pred(input_node);

                        input_node->get_rt_info()["depth"] = is_valid ? 0 : cur_depth + 1;
                        input_node->get_rt_info()["used"] = 1;

                        set_node_type(input_node);
                        q.push_front({input_node, cur, false});
                    } else {
                        graph.value[Graph::Vertex(cur.get())].insert(Graph::Vertex(input_node.get()));
                    }
                }
            }
        }
    };

    auto g = Graph();
    for (auto & node : nodes) {
        if (is_used(node)) continue;
        traverse(node, g);
    }

    return g;
}

Graph combine_graphs(const std::vector<Graph> & graphs) {
    assert(!graphs.empty());
    Graph g(graphs[0]);
    for (size_t i = 1; i < graphs.size(); ++i) {
        for (const auto & item : graphs[i].value) {
            if (!g.value.count(item.first)) {
                g.value[item.first] = {};
            }
            g.value[item.first].insert(item.second.begin(), item.second.end());
        }
    }
    return g;
}

// ------------------------------ Colored RT attribute -------------------------------------------

class Colored : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("colored", "0");

    Colored() = default;

    bool is_copyable() const override {
        return false;
    }
};

template <typename T>
void set_is_colored(T node, size_t id) {
    const std::string key = "colored_" + std::to_string(id);
    node->get_rt_info()[key] = Colored();
}

template <typename T>
bool is_colored(T node, size_t id) {
    const std::string key = "colored_" + std::to_string(id);
    return node->get_rt_info().count(key);
}

// ------------------------------ PassTracker definitions -------------------------------------------

ov::pass::PassTrackerState ov::pass::PassTracker::m_state = ov::pass::PassTrackerState();

ov::pass::PassTracker::PassTracker(std::shared_ptr<ov::Model> orig_model,
                                   std::string pass_name)
    : m_orig_model(std::move(orig_model))
    , m_pass_name(std::move(pass_name)) {
    bool pass_tracking_enabled = ov::util::getenv_bool("NGRAPH_ENABLE_PASS_TRACKING");
    if (!pass_tracking_enabled || !m_state.is_enabled()) return;

    if (m_state.is_active()) {
        // Check if tracker state was already acquired by some other tracker (which means that
        // this pass is a nested pass), so we skip current tracker in favour of global one.
        m_skip_tracker = true;
        // std::cout << "[ DEBUG : PassTracker ] skip pass tracker for " << m_pass_name << " as the state was already acquired by " << m_state.m_pass_name << std::endl;
        return;
    } else {
        // Acquire tracker state.
        m_state.set_is_active(true);
        m_state.set_pass_name(m_pass_name);
        // std::cout << "[ DEBUG : PassTracker ] pass tracker state was acquired by " << m_pass_name << std::endl;
    }

    m_local_pass_tracker_id = m_state.get_pass_tracker_id();

    ov::NodeVector orig_nodes = m_orig_model->get_ordered_ops();

    // Markup original model operations with rt info attribute, so we can easily
    // detect new nodes (they won't have this attribute).
    for (auto &orig_node : orig_nodes) {
        set_is_colored(orig_node, m_local_pass_tracker_id);
    }

    // If tracker state was reset or we execute PassTracker for the first time, so we make
    // a model copy, so we can track two model states (before transformation, and after).
    // I also used a notation that `model` - represents model before transformation which we
    // update between passes, and `orig_model` - an actual model that is being transformed.
    if (!m_state.model()) {
        auto model = ov::clone_model(*m_orig_model);
        m_state.set_model(model);

        ov::NodeVector cloned_nodes = model->get_ordered_ops();

        size_t i{0};
        // Create initial mapping between model and original_model, so we can easily detect which nodes were removed
        // after transformation applied, and it also helps to update model to original_model without copy.
        for (auto & orig_node : orig_nodes) {
            assert(cloned_nodes[i]->get_type_info() == orig_node->get_type_info());
            // TODO: can cause a problem if topological sorts are not identical.
            m_state.model_to_orig()[cloned_nodes[i]] = orig_node;
            cloned_nodes[i]->set_instance_id(orig_node->get_instance_id());
            ++i;
        }
    }
}

ov::pass::PassTracker::~PassTracker() {
    bool pass_tracking_enabled = ov::util::getenv_bool("NGRAPH_ENABLE_PASS_TRACKING");
    if (!pass_tracking_enabled || m_skip_tracker || !m_state.is_enabled()) return;

    auto clean_rt_attrs = [](std::shared_ptr<ov::Node> node) {
        if (!node) return;
        auto &rt_info = node->get_rt_info();
        rt_info.erase("used");
        rt_info.erase("depth");
        rt_info.erase("vertex_type");
        rt_info.erase("eliminated");
    };

    auto & model_to_orig = m_state.model_to_orig();

    // Clean rt info attributes that remain from the previous iteration to avoid any conflicts.
    for (auto & item : model_to_orig) {
        clean_rt_attrs(item.first);
        clean_rt_attrs(item.second.lock());
    }

    // Detect nodes that were created during transformation by checking `color` rt attribute.
    ov::NodeVector created_nodes;
    for (auto &item : m_orig_model->get_ordered_ops()) {
        if (!is_colored(item, m_local_pass_tracker_id)) {
            created_nodes.push_back(item);
        }
    }

    // Creates a graph which contain sub-graph(s) of nodes that were created during
    // transformation + some nodes around them (context).
    auto created_graph = model_partitioning(created_nodes, [=](std::shared_ptr<ov::Node> tmp) -> bool {
        return !is_colored(tmp, m_local_pass_tracker_id);
    }, "created");

    // Mapping from the nodes in original model to its previous state. This is used later to
    // update `model` without full `original_model` copy.
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> orig_to_model;

    // Detect nodes that were removed during transformation by checking weak pointers to the nodes
    // in original model. If it is nullptr, so the node was removed.
    ov::NodeVector eliminated_nodes;
    for (auto &item : model_to_orig) {
        if (!item.second.lock()) {
            eliminated_nodes.push_back(item.first);
            item.first->get_rt_info()["eliminated"] = 1;
        } else {
            orig_to_model[item.second.lock()] = item.first;
        }
    }

    // Remove eliminated nodes from mapping.
    for (auto & node : eliminated_nodes) {
        model_to_orig.erase(node);
    }

    // Creates a graph which contain sub-graph(s) of nodes that were eliminated during
    // transformation + some nodes around them (context).
    auto eliminated_graph = model_partitioning(eliminated_nodes, [](std::shared_ptr<ov::Node> tmp) -> bool {
        return tmp->get_rt_info().count("eliminated") != 0;
    }, "eliminated");

    // Creates a graph which combines both types of nodes (created and eliminated).
    auto combined_graph = combine_graphs({eliminated_graph, created_graph});

    // Graphs to JSON serialization.
    const std::vector<std::pair<std::string, std::function<bool(const Graph::Vertex &)>>> colors = {
            {"orange", [](const Graph::Vertex &v) { return v.vertex_type == "eliminated"; }},
            {"green",  [](const Graph::Vertex &v) { return v.vertex_type == "created"; }},
    };

    static size_t index{0};
    graph_to_json(created_graph, m_pass_name, index, "created", colors);
    graph_to_json(eliminated_graph, m_pass_name, index, "eliminated", colors);
    graph_to_json(combined_graph, m_pass_name, index, "combined", colors);

    // This function updates `model` to be aligned with `original_model` without its full copy.
    auto update_function = [&](std::shared_ptr<ov::Model> model, std::shared_ptr<ov::Model> orig_model) {
        // TODO: add description.
        for (auto & orig_node : orig_model->get_ordered_ops()) {
            ov::OutputVector outputs;
            for (auto & orig_output : orig_node->input_values()) {
                auto orig_input_node = orig_output.get_node_shared_ptr();
                assert(orig_to_model.count(orig_input_node));

                outputs.push_back(
                    orig_to_model[orig_input_node]->output(orig_output.get_index())
                );
            }

            if (orig_to_model.count(orig_node) == 0) {
                // TODO: check if node is parameter/result.
                auto node = orig_node->clone_with_new_inputs(outputs);
                node->set_instance_id(orig_node->get_instance_id());
                orig_to_model[orig_node] = node;
                model_to_orig[node] = orig_node;
            } else {
                auto node = orig_to_model[orig_node];
                size_t input_id = 0;
                for (auto & output : node->input_values()) {
                    // TODO: verify that this check works.
                    if (output != outputs[input_id]) {
                        node->input(input_id).replace_source_output(outputs[input_id]);
                    }
                    input_id++;
                }
            }
        }
    };

    // Update `model` only in case if something has changed in original model.
    if (!created_graph.is_empty() || !eliminated_graph.is_empty() || !created_graph.is_empty()) {
        update_function(m_state.model(), m_orig_model);
    }

    // Release tracker state.
    m_state.set_is_active(false);

    index++;
}