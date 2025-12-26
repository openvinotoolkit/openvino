
graph_info_dot = """\
digraph {
    {%- for node_id, node in deps_graph["nodes"].items() %}
        {%- for dep_id, dep in node["dependencies"].items() %}
        {%- if dep["direct"] %}
        "{{ node["label"] }}" -> "{{ deps_graph["nodes"][dep_id]["label"] }}"
        {%- endif %}
        {%- endfor %}
    {%- endfor %}
}

"""
