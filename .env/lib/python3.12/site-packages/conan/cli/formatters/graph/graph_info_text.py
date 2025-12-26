import fnmatch
from collections import OrderedDict

from conan.api.model import RecipeReference
from conan.api.output import ConanOutput, cli_out_write


def filter_graph(graph, package_filter=None, field_filter=None):
    if package_filter is not None:
        def _matching(node, pattern):
            if fnmatch.fnmatch(node["ref"] or "", pattern):
                return True
            if pattern == "&":  # Handle the consumer pattern
                if node["recipe"] == "Consumer":
                    return True
                # How to deal with --requires=xxx --package-filter=& "consumers"
                root = graph["nodes"]["0"]
                if root["recipe"] == "Cli" and node is not root:
                    # We look if the current node is a direct dependency of the root node
                    node_ref = RecipeReference.loads(node["ref"])
                    for dep in root["dependencies"].values():
                        if dep["direct"] and node_ref == RecipeReference.loads(dep["ref"]):
                            return True

        graph["nodes"] = {id_: n for id_, n in graph["nodes"].items()
                          if any(_matching(n, p) for p in package_filter)}
    if field_filter is not None:
        if "ref" not in field_filter:
            field_filter.append("ref")
        result = {}
        for id_, n in graph["nodes"].items():
            new_node = OrderedDict((k, v) for k, v in n.items() if k in field_filter)
            result[id_] = new_node
        graph["nodes"] = result
    return graph


def format_graph_info(result):
    """ More complete graph output, including information for every node in the graph
    Used for 'graph info' command
    """
    graph = result["graph"]
    field_filter = result["field_filter"]
    package_filter = result["package_filter"]

    ConanOutput().subtitle("Basic graph information")
    serial = graph.serialize()
    serial = filter_graph(serial, package_filter, field_filter)
    for n in serial["nodes"].values():
        cli_out_write(f"{n['ref']}:")  # FIXME: This can be empty for consumers and it is ugly ":"
        _serial_pretty_printer(n, indent="  ")


def _serial_pretty_printer(data, indent=""):
    for k, v in data.items():
        if isinstance(v, dict):
            cli_out_write(f"{indent}{k}:")
            # TODO: increment color too
            _serial_pretty_printer(v, indent=indent+"  ")
        else:
            cli_out_write(f"{indent}{k}: {v}")
