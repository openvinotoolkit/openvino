from collections import OrderedDict

from conan.api.model import RecipeReference
from conan.errors import ConanException
from conan.internal.graph.graph import RECIPE_PLATFORM
from conan.internal.model.conanfile_interface import ConanFileInterface


class UserRequirementsDict(object):
    """ user facing dict to allow access of dependencies by name
    """
    def __init__(self, data, require_filter=None):
        self._data = data  # dict-like
        self._require_filter = require_filter  # dict {trait: value} for requirements

    def filter(self, require_filter):
        def filter_fn(require):
            for k, v in require_filter.items():
                if getattr(require, k) != v:
                    return False
            return True
        data = OrderedDict((k, v) for k, v in self._data.items() if filter_fn(k))
        return UserRequirementsDict(data, require_filter)

    def __bool__(self):
        return bool(self._data)

    def get(self, ref, build=None, **kwargs):
        return self._get(ref, build, **kwargs)[1]

    def _get(self, ref, build=None, **kwargs):
        if build is None:
            current_filters = self._require_filter or {}
            if "build" not in current_filters:
                # By default we search in the "host" context
                kwargs["build"] = False
        else:
            kwargs["build"] = build
        data = self.filter(kwargs)
        ret = []
        if "/" in ref:
            # FIXME: Validate reference
            ref = RecipeReference.loads(ref)
            for require, value in data.items():
                if require.ref == ref:
                    ret.append((require, value))
        else:
            name = ref
            for require, value in data.items():
                if require.ref.name == name:
                    ret.append((require, value))
        if len(ret) > 1:
            current_filters = data._require_filter or "{}"
            requires = "\n".join(["- {}".format(require) for require, _ in ret])
            raise ConanException("There are more than one requires matching the specified filters:"
                                 " {}\n{}".format(current_filters, requires))
        if not ret:
            raise KeyError("'{}' not found in the dependency set".format(ref))

        key, value = ret[0]
        return key, value

    def __getitem__(self, name):
        return self.get(name)

    def __delitem__(self, name):
        r, _ = self._get(name)
        del self._data[r]

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __contains__(self, item):
        try:
            self.get(item)
            return True
        except KeyError:
            return False
        except ConanException:
            # ConanException is raised when there are more than one matching the filters
            # so it's definitely in the dict
            return True


class ConanFileDependencies(UserRequirementsDict):

    @staticmethod
    def from_node(node):
        d = OrderedDict((require, ConanFileInterface(transitive.node.conanfile))
                        for require, transitive in node.transitive_deps.items())
        if node.replaced_requires:
            cant_be_removed = set()
            for old_req, new_req in node.replaced_requires.items():
                # Two different replaced_requires can point to the same real requirement
                existing = d[new_req]
                added_req = new_req.copy_requirement()
                added_req.ref = RecipeReference.loads(old_req)
                d[added_req] = existing
                if new_req.ref.name == added_req.ref.name:
                    cant_be_removed.add(new_req)
            # Now remove the replaced from dependencies dict
            for new_req in node.replaced_requires.values():
                if new_req not in cant_be_removed:
                    d.pop(new_req, None)
        return ConanFileDependencies(d)

    def filter(self, require_filter, remove_system=True):
        # FIXME: Copy of hte above, to return ConanFileDependencies class object
        def filter_fn(require):
            for k, v in require_filter.items():
                if getattr(require, k) != v:
                    return False
            return True

        data = OrderedDict((k, v) for k, v in self._data.items() if filter_fn(k))
        if remove_system:
            data = OrderedDict((k, v) for k, v in data.items() if v.recipe != RECIPE_PLATFORM)
        return ConanFileDependencies(data, require_filter)

    def transitive_requires(self, other):
        """
        :type other: ConanFileDependencies
        """
        data = OrderedDict()
        for k, v in self._data.items():
            for otherk, otherv in other._data.items():
                if v == otherv:
                    data[otherk] = v  # Use otherk to respect original replace_requires
        return ConanFileDependencies(data)

    @property
    def topological_sort(self):
        # Return first independent nodes, final ones are the more direct deps
        result = OrderedDict()
        opened = self._data.copy()

        while opened:
            opened_values = set(opened.values())
            new_opened = OrderedDict()
            for req, conanfile in opened.items():
                deps_in_opened = any(d in opened_values for d in conanfile.dependencies.values())
                if deps_in_opened:
                    new_opened[req] = conanfile  # keep it for next iteration
                else:
                    result[req] = conanfile  # No dependencies in open set!

            opened = new_opened
        return ConanFileDependencies(result)

    @property
    def direct_host(self):
        return self.filter({"build": False, "direct": True, "test": False, "skip": False})

    @property
    def direct_build(self):
        return self.filter({"build": True, "direct": True})

    @property
    def host(self):
        return self.filter({"build": False, "test": False, "skip": False})

    @property
    def test(self):
        # Not needed a direct_test because they are visible=False so only the direct consumer
        # will have them in the graph
        return self.filter({"build": False, "test": True, "skip": False})

    @property
    def build(self):
        return self.filter({"build": True})


def get_transitive_requires(consumer, dependency):
    """ the transitive requires that we need are the consumer ones, not the current dependencey
    ones, so we get the current ones, then look for them in the consumer, and return those
    """
    # The build dependencies cannot be transitive in generators like CMakeDeps,
    # even if users make them visible
    pkg_deps = dependency.dependencies.filter({"direct": True, "build": False})
    # First we filter the skipped dependencies
    result = consumer.dependencies.filter({"skip": False})
    # and we keep those that are really dependencies of the current package
    result = result.transitive_requires(pkg_deps)
    return result
