from conan.internal.graph.graph_error import GraphProvidesError
from conan.api.model import RecipeReference


def check_graph_provides(dep_graph):
    if dep_graph.error:
        return
    for node in dep_graph.nodes:
        provides = {}
        current_provides = node.conanfile.provides
        if isinstance(current_provides, str):  # Just in case it is defined in configure() as str
            current_provides = [current_provides]
        for dep in node.transitive_deps.values():
            dep_node = dep.node
            if node.ref is not None and dep_node.ref.name == node.ref.name:
                continue  # avoid dependency to self (as tool-requires for cross-build)

            dep_provides = dep_node.conanfile.provides
            if dep_provides is None:
                continue
            if isinstance(dep_provides, str):
                dep_provides = [dep_provides]
            for provide in dep_provides:
                # First check if collides with current node
                if current_provides is not None and provide in current_provides:
                    raise GraphProvidesError(node, dep_node)

                # Then, check if collides with other requirements
                new_req = dep.require.copy_requirement()
                new_req.ref = RecipeReference(provide, new_req.ref.version, new_req.ref.user,
                                              new_req.ref.channel)
                existing = node.transitive_deps.get(new_req)
                if existing is not None:
                    raise GraphProvidesError(existing.node, dep_node)
                else:
                    existing_provide = provides.get(new_req)
                    if existing_provide is not None:
                        raise GraphProvidesError(existing_provide, dep_node)
                    else:
                        provides[new_req] = dep_node
