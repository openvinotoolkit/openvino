from collections import OrderedDict

from conan.internal.graph.graph_error import GraphError, GraphConflictError
from conan.api.output import ConanOutput
from conan.api.model import PkgReference
from conan.api.model import RecipeReference

RECIPE_DOWNLOADED = "Downloaded"
RECIPE_INCACHE = "Cache"  # The previously installed recipe in cache is being used
RECIPE_UPDATED = "Updated"
RECIPE_INCACHE_DATE_UPDATED = "Cache (Updated date)"
RECIPE_NEWER = "Newer"  # The local recipe is  modified and newer timestamp than server
RECIPE_NOT_IN_REMOTE = "Not in remote"
RECIPE_UPDATEABLE = "Update available"  # The update of recipe is available (only in conan info)
RECIPE_NO_REMOTE = "No remote"
RECIPE_EDITABLE = "Editable"
RECIPE_CONSUMER = "Consumer"  # A conanfile from the user
RECIPE_VIRTUAL = "Cli"  # A virtual conanfile (dynamic in memory conanfile)
RECIPE_PLATFORM = "Platform"

BINARY_CACHE = "Cache"
BINARY_DOWNLOAD = "Download"
BINARY_UPDATE = "Update"
BINARY_BUILD = "Build"
BINARY_MISSING = "Missing"
BINARY_SKIP = "Skip"
BINARY_EDITABLE = "Editable"
BINARY_EDITABLE_BUILD = "EditableBuild"
BINARY_INVALID = "Invalid"
BINARY_PLATFORM = "Platform"

CONTEXT_HOST = "host"
CONTEXT_BUILD = "build"


class TransitiveRequirement:
    def __init__(self, require, node):
        self.require = require
        self.node = node

    def __repr__(self):
        return "Require: {}, Node: {}".format(repr(self.require), repr(self.node))


class Node:

    def __init__(self, ref, conanfile, context, recipe=None, path=None, test=False):
        self.ref = ref
        self.path = path  # path to the consumer conanfile.xx for consumer, None otherwise
        self._package_id = None
        self.prev = None
        self.pref_timestamp = None
        if conanfile is not None:
            conanfile._conan_node = self  # Reference to self, to access data
        self.conanfile = conanfile

        self.binary = None
        self.recipe = recipe
        self.remote = None
        self.binary_remote = None
        self.context = context
        self.test = test

        # real graph model
        self.transitive_deps = OrderedDict()  # of _TransitiveRequirement
        self.edges = []  # Ordered Edges
        self.dependants = []  # Edges
        self.error = None
        self.should_build = False  # If the --build or policy wants to build this binary
        self.build_allowed = False
        self.is_conf = False
        self.replaced_requires = {}  # To track the replaced requires for self.edges[old-ref]
        self.skipped_build_requires = False

    @property
    def dependencies(self):
        ConanOutput().warning("Node.dependencies is private and shouldn't be used. It is now "
                              "node.edges. Please fix your code, Node.dependencies will be removed "
                              "in future versions", warn_tag="deprecated")
        return self.edges

    def subgraph(self):
        nodes = [self]
        opened = [self]
        while opened:
            new_opened = []
            for o in opened:
                for n in o.neighbors():
                    if n not in nodes:
                        nodes.append(n)
                    if n not in opened:
                        new_opened.append(n)
            opened = new_opened

        graph = DepsGraph()
        graph.nodes = nodes
        return graph

    def __lt__(self, other):
        """
        :type other: Node
        """
        # TODO: Remove this order, shouldn't be necessary
        return (str(self.ref), self._package_id) < (str(other.ref), other._package_id)

    def propagate_closing_loop(self, require, prev_node, visibility_conflicts):
        self.propagate_downstream(require, prev_node, visibility_conflicts)
        # List to avoid mutating the dict
        for transitive in list(prev_node.transitive_deps.values()):
            # TODO: possibly optimize in a bulk propagate
            if transitive.require.override:
                continue
            prev_node.propagate_downstream(transitive.require, transitive.node, visibility_conflicts,
                                           self)

    def propagate_downstream(self, require, node, visibility_conflicts, src_node=None):
        # print("  Propagating downstream ", self, "<-", require)
        assert node is not None
        # This sets the transitive_deps node if it was None (overrides)
        # Take into account that while propagating we can find RUNTIME shared conflicts we
        # didn't find at check_downstream_exist, because we didn't know the shared/static
        existing = self.transitive_deps.get(require)
        ill_formed = False
        if existing is not None and existing.require is not require:
            if existing.node is not None and existing.node.ref != node.ref:
                # print("  +++++Runtime conflict!", require, "with", node.ref)
                raise GraphConflictError(self, require, existing.node, existing.require, node)
            ill_formed = ((require.direct or existing.require.direct)
                          and require.visible != existing.require.visible)
            if ill_formed and not (require.test or existing.require.test):
                visibility_conflicts.setdefault(require.ref, set()).add(self.ref)
            require.aggregate(existing.require)
            # An override can be overriden by a downstream force/override
            if existing.require.override and existing.require.ref != require.ref:
                # If it is an override, but other value, it has been overriden too
                existing.require.overriden_ref = existing.require.ref
                existing.require.override_ref = require.ref

        assert not require.version_range  # No ranges slip into transitive_deps definitions
        # TODO: Might need to move to an update() for performance
        poped = self.transitive_deps.pop(require, None)
        self.transitive_deps[require] = TransitiveRequirement(require, node)
        if poped is not None:  # adjust .edges, to avoid orphans
            for e in self.edges:
                if e.dst is poped.node:  # check for identity, pointing to that node
                    e.dst = node
                    break

        if self.conanfile.vendor:
            return
        # Check if need to propagate downstream
        if not self.dependants:
            return

        if src_node is not None:  # This happens when closing a loop, and we need to know the edge
            d = next(d for d in self.dependants if d.src is src_node)
        else:
            assert len(self.dependants) == 1
            d = self.dependants[0]

        down_require = d.require.transform_downstream(self.conanfile.package_type, require,
                                                      node.conanfile.package_type)
        if down_require is None:
            return

        down_require.defining_require = require.defining_require
        # If the requirement propagates .files downstream, cannot be skipped
        # But if the files are not needed in this graph branch, can be marked "Skip"
        if down_require.files:
            down_require.required_nodes = require.required_nodes.copy()
        down_require.required_nodes.add(self)
        d.src.propagate_downstream(down_require, node, visibility_conflicts)

    def check_downstream_exists(self, require):
        # First, a check against self, could be a loop-conflict
        # This is equivalent as the Requirement hash and eq methods
        # TODO: Make self.ref always exist, but with name=None if name not defined
        if self.ref is not None and require.ref.name == self.ref.name:
            if require.build and (self.context == CONTEXT_HOST or  # switch context
                                  require.ref.version != self.ref.version):  # or different version
                pass
            elif require.visible is False:  # and require.ref.version != self.ref.version:
                # Experimental, to support repackaging of openssl previous versions FIPS plugins
                pass  # An invisible require doesn't conflict with itself
            else:
                return None, self, self  # First is the require, as it is a loop => None

        # First do a check against the current node dependencies
        prev = self.transitive_deps.get(require)
        # print("    Transitive deps", self.transitive_deps)
        # ("    THERE IS A PREV ", prev, "in node ", self, " for require ", require)
        # Overrides: The existing require could be itself, that was just added
        result = None
        if prev and (prev.require is not require or prev.node is not None):
            result = prev.require, prev.node, self
            # Do not return yet, keep checking downstream, because downstream overrides or forces
            # have priority

        # Check if need to propagate downstream
        # Then propagate downstream

        if self.conanfile.vendor:
            return result
        # Seems the algrithm depth-first, would only have 1 dependant at most to propagate down
        # at any given time
        if not self.dependants:
            return result
        assert len(self.dependants) == 1
        dependant = self.dependants[0]

        # TODO: Implement an optimization where the requires is checked against a graph global
        # print("    Lets check_downstream one more")
        down_require = dependant.require.transform_downstream(self.conanfile.package_type,
                                                              require, None)

        if down_require is None:
            # print("    No need to check downstream more")
            return result

        down_require.defining_require = require.defining_require
        source_node = dependant.src
        return source_node.check_downstream_exists(down_require) or result

    def check_loops(self, new_node, count=0):
        if self.ref == new_node.ref and self.context == new_node.context:
            if count >= 1:
                return self
            count += 1
        if not self.dependants:
            return
        assert len(self.dependants) == 1
        dependant = self.dependants[0]
        source_node = dependant.src
        return source_node.check_loops(new_node, count)

    @property
    def package_id(self):
        return self._package_id

    @package_id.setter
    def package_id(self, pkg_id):
        assert self._package_id is None, "Trying to override an existing package_id"
        self._package_id = pkg_id

    @property
    def name(self):
        return self.ref.name if self.ref else None

    @property
    def pref(self):
        assert self.ref is not None and self.package_id is not None, "Node %s" % self.recipe
        return PkgReference(self.ref, self.package_id, self.prev, self.pref_timestamp)

    def add_edge(self, edge):
        if edge.src == self:
            self.edges.append(edge)
        else:
            self.dependants.append(edge)

    def neighbors(self):
        return [edge.dst for edge in self.edges]

    def inverse_neighbors(self):
        return [edge.src for edge in self.dependants]

    def __repr__(self):
        return repr(self.conanfile)

    def serialize(self):
        result = OrderedDict()
        result["ref"] = self.ref.repr_notime() if self.ref is not None else "conanfile"
        result["id"] = getattr(self, "id")  # Must be assigned by graph.serialize()
        result["recipe"] = self.recipe
        result["package_id"] = self.package_id
        result["prev"] = self.prev
        result["rrev"] = self.ref.revision if self.ref is not None else None
        result["rrev_timestamp"] = self.ref.timestamp if self.ref is not None else None
        result["prev_timestamp"] = self.pref_timestamp
        result["remote"] = self.remote.name if self.remote else None
        result["binary_remote"] = self.binary_remote.name if self.binary_remote else None
        from conan.internal.graph.installer import build_id
        result["build_id"] = build_id(self.conanfile)
        result["binary"] = self.binary
        # TODO: This doesn't match the model, check it
        result["invalid_build"] = getattr(getattr(self.conanfile, "info", None), "cant_build", False)
        result["info_invalid"] = getattr(getattr(self.conanfile, "info", None), "invalid", None)
        # Adding the conanfile information: settings, options, etc
        result.update(self.conanfile.serialize())
        result.pop("requires", None)  # superseded by "dependencies" (graph.transitive_deps)
        result["dependencies"] = {d.node.id: d.require.serialize()
                                  for d in self.transitive_deps.values() if d.node is not None}
        result["context"] = self.context
        result["test"] = self.test
        return result

    def overrides(self):

        def transitive_subgraph():
            result = set()
            opened = {self}
            while opened:
                new_opened = set()
                for o in opened:
                    result.add(o)
                    new_opened.update(set(o.neighbors()).difference(result))
                opened = new_opened

            return result

        nodes = transitive_subgraph()
        return Overrides.create(nodes)


class Edge:
    def __init__(self, src, dst, require):
        self.src = src
        self.dst = dst
        self.require = require


class Overrides:
    def __init__(self):
        self._overrides = {}  # {require_ref: {override_ref1, override_ref2}}

    def __bool__(self):
        return bool(self._overrides)

    def __repr__(self):
        return repr(self.serialize())

    @staticmethod
    def create(nodes):
        overrides = {}
        for n in nodes:
            for r in n.conanfile.requires.values():
                if r.override and not r.overriden_ref:  # overrides are not real graph edges
                    continue
                if r.overriden_ref:
                    overrides.setdefault(r.overriden_ref, set()).add(r.override_ref)
                else:
                    overrides.setdefault(r.ref, set()).add(None)

        # reduce, eliminate those overrides definitions that only override to None, that is, not
        # really an override
        result = Overrides()
        for require, override_info in overrides.items():
            if len(override_info) != 1 or None not in override_info:
                result._overrides[require] = override_info
        return result

    def get(self, require):
        return self._overrides.get(require)

    def update(self, other):
        """
        :type other: Overrides
        """
        for require, override_info in other._overrides.items():
            self._overrides.setdefault(require, set()).update(override_info)

    def items(self):
        return self._overrides.items()

    def serialize(self):
        return {k.repr_notime(): sorted([e.repr_notime() if e else None for e in v],
                                        key=lambda e: "" if e is None else e)
                for k, v in self._overrides.items()}

    @staticmethod
    def deserialize(data):
        result = Overrides()
        result._overrides = {RecipeReference.loads(k):
                             set([RecipeReference.loads(e) if e else None for e in v])
                             for k, v in data.items()}
        return result


class DepsGraph:
    def __init__(self):
        self.nodes = []
        self.aliased = {}
        self.resolved_ranges = {}
        self.replaced_requires = {}
        self.options_conflicts = {}
        self.visibility_conflicts = {}
        self.error = False

    def lockfile(self):
        from conan.internal.model.lockfile import Lockfile
        return Lockfile(self)

    def overrides(self):
        return Overrides.create(self.nodes)

    def __repr__(self):
        return "\n".join((repr(n) for n in self.nodes))

    @property
    def root(self):
        return self.nodes[0] if self.nodes else None

    def add_node(self, node):
        self.nodes.append(node)

    @staticmethod
    def add_edge(src, dst, require):
        edge = Edge(src, dst, require)
        src.add_edge(edge)
        dst.add_edge(edge)

    def ordered_iterate(self):
        ordered = self.by_levels()
        for level in ordered:
            for node in level:
                yield node

    def by_levels(self):
        """ order by node degree. The first level will be the one which nodes dont have
        dependencies. Second level will be with nodes that only have dependencies to
        first level nodes, and so on
        return [[node1, node34], [node3], [node23, node8],...]
        """
        result = []
        # We make it a dict to preserve insertion order and be deterministic, s
        # sets are not deterministic order. dict is fast for look up operations
        opened = dict.fromkeys(self.nodes)
        while opened:
            current_level = []
            for o in opened:
                o_neighs = o.neighbors()
                if not any(n in opened for n in o_neighs):
                    current_level.append(o)

            # TODO: SORTING seems only necessary for test order
            current_level.sort()
            result.append(current_level)
            # now start new level, removing the current level items
            for item in current_level:
                opened.pop(item)

        return result

    def report_graph_error(self):
        if self.error:
            raise self.error

    def serialize(self):
        for i, n in enumerate(self.nodes):
            n.id = str(i)
        result = OrderedDict()
        result["nodes"] = {n.id: n.serialize() for n in self.nodes}
        result["root"] = {self.root.id: repr(self.root.ref)}  # TODO: ref of consumer/virtual
        result["overrides"] = self.overrides().serialize()
        result["resolved_ranges"] = {repr(r): s.repr_notime()
                                     for r, s in self.resolved_ranges.items()}
        result["replaced_requires"] = {k: v for k, v in self.replaced_requires.items()}
        result["error"] = self.error.serialize() if isinstance(self.error, GraphError) else None
        return result
