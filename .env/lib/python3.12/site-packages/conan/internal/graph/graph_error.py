from conan.errors import ConanException


class GraphError(ConanException):
    def serialize(self):
        return


class GraphConflictError(GraphError):

    def __init__(self, node, require, prev_node, prev_require, base_previous):
        self.node = node
        self.require = require
        self.prev_node = prev_node
        self.prev_require = prev_require
        self.base_previous = base_previous

    def serialize(self):
        dst_id = self.prev_node.id if self.prev_node else None
        return {"type": "conflict",
                "name": self.require.ref.name,
                "branch1": {"src_id": self.base_previous.id, "src_ref": str(self.base_previous.ref),
                            "dst_id": dst_id, "require": self.prev_require.serialize()},
                "branch2": {"src_id": self.node.id, "src_ref": str(self.node.ref),
                            "require": self.require.serialize()}}

    def __str__(self):
        conflicting_node = self.node.ref or self.base_previous.ref
        conflicting_node_msg = ""
        if conflicting_node is not None:
            conflicting_node_msg = f"\nConflict originates from {conflicting_node}\n"
        return f"Version conflict: " \
               f"Conflict between {self.require.ref} and {self.prev_require.ref} in the graph." \
               f"{conflicting_node_msg}" \
               f"\nRun 'conan graph info ... --format=html > graph.html' " \
               f"and open 'graph.html' to inspect the conflict graphically."


class GraphLoopError(GraphError):

    def __init__(self, node, require, ancestor):
        self.node = node
        self.require = require
        self.ancestor = ancestor

    def __str__(self):
        return "There is a cycle/loop in the graph:\n" \
               f"    Initial ancestor: {self.ancestor}\n" \
               f"    Require: {self.require.ref}\n" \
               f"    Dependency: {self.node}"


class GraphMissingError(GraphError):

    def __init__(self, node, require, missing_error):
        self.node = node
        self.require = require
        self.missing_error = missing_error

    def serialize(self):
        return {"type": "missing",
                "node": {"id": self.node.id, "ref": str(self.node.ref)},
                "require": self.require.serialize(),
                "error": self.missing_error}

    def __str__(self):
        return (f"Package '{self.require.ref}' not resolved: {self.missing_error}. "
                f"Required by '{self.node.ref or 'cli'}'")


class GraphProvidesError(GraphError):

    def __init__(self, node, conflicting_node):
        self.node = node
        self.conflicting_node = conflicting_node
        node.error = conflicting_node.error

    def serialize(self):
        return {"type": "provide_conflict",
                "node": {"id": self.node.id, "ref": str(self.node.ref)},
                "conflicting_node": {"id": self.conflicting_node.id,
                                     "ref": str(self.conflicting_node.ref)},
                "provided": self.node.conanfile.provides or self.conflicting_node.conanfile.provides}

    def __str__(self):
        provides = self.node.conanfile.provides or self.conflicting_node.conanfile.provides
        return f"Provide Conflict: Both '{self.node.ref}' and '{self.conflicting_node.ref}' " \
               f"provide '{provides}'."
