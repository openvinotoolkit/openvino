import os

from conan.errors import ConanException
from conan.api.model import RecipeReference
from conan.internal.model.requires import Requirement


class PyRequire:
    def __init__(self, module, conanfile, ref, path, recipe_status, remote):
        self.module = module
        self.conanfile = conanfile
        self.ref = ref
        self.path = path
        self.recipe = recipe_status
        self.remote = remote

    def serialize(self):
        return {"remote": self.remote.name if self.remote is not None else None,
                "recipe": self.recipe,
                "path": self.path}


class PyRequires:
    """ this is the object that replaces the declared conanfile.py_requires"""
    def __init__(self):
        self._pyrequires = {}  # {pkg-name: PythonRequire}

    def serialize(self):
        return {r.ref.repr_notime(): r.serialize() for r in self._pyrequires.values()}

    def all_refs(self):
        return [r.ref for r in self._pyrequires.values()]

    def info_requires(self):
        return {pyreq.ref: getattr(pyreq.conanfile, "package_id_python_mode", None)
                for pyreq in self._pyrequires.values()}

    def items(self):
        return self._pyrequires.items()

    def __getitem__(self, item):
        try:
            return self._pyrequires[item]
        except KeyError:
            raise ConanException("'%s' is not a python_require" % item)

    def add_pyrequire(self, py_require):
        key = py_require.ref.name
        # single item assignment, direct
        existing = self._pyrequires.get(key)
        if existing and existing is not py_require:  # if is the same one, can be added.
            # TODO: Better test python_requires conflicts
            raise ConanException("The python_require '%s' already exists" % key)
        self._pyrequires[key] = py_require

        transitive = getattr(py_require.conanfile, "python_requires", None)
        if transitive is None:
            return
        for name, transitive_py_require in transitive.items():
            existing = self._pyrequires.get(name)
            if existing and existing.ref != transitive_py_require.ref:
                raise ConanException("Conflict in py_requires %s - %s"
                                     % (existing.ref, transitive_py_require.ref))
            self._pyrequires[name] = transitive_py_require


class PyRequireLoader(object):
    def __init__(self, conan_app, global_conf):
        self._proxy = conan_app.proxy
        self._range_resolver = conan_app.range_resolver
        self._cached_py_requires = {}
        self._resolve_prereleases = global_conf.get("core.version_ranges:resolve_prereleases")

    def load_py_requires(self, conanfile, loader, graph_lock, remotes, update, check_update):
        py_requires_refs = getattr(conanfile, "python_requires", None)
        if py_requires_refs is None:
            return
        if isinstance(py_requires_refs, str):
            py_requires_refs = [py_requires_refs, ]

        py_requires = self._resolve_py_requires(py_requires_refs, graph_lock, loader, remotes,
                                                update, check_update)
        if hasattr(conanfile, "python_requires_extend"):
            py_requires_extend = conanfile.python_requires_extend
            if isinstance(py_requires_extend, str):
                py_requires_extend = [py_requires_extend, ]
            for p in py_requires_extend:
                pkg_name, base_class_name = p.rsplit(".", 1)
                base_class = getattr(py_requires[pkg_name].module, base_class_name)
                conanfile.__bases__ = (base_class,) + conanfile.__bases__
        conanfile.python_requires = py_requires

    def _resolve_py_requires(self, py_requires_refs, graph_lock, loader, remotes, update,
                             check_update):
        result = PyRequires()
        for py_requires_ref in py_requires_refs:
            py_requires_ref = RecipeReference.loads(py_requires_ref)
            requirement = Requirement(py_requires_ref)
            resolved_ref = self._resolve_ref(requirement, graph_lock, remotes, update)
            try:
                py_require = self._cached_py_requires[resolved_ref]
            except KeyError:
                pyreq_conanfile = self._load_pyreq_conanfile(loader, graph_lock, resolved_ref,
                                                             remotes, update, check_update)
                conanfile, module, new_ref, path, recipe_status, remote = pyreq_conanfile
                py_require = PyRequire(module, conanfile, new_ref, path, recipe_status, remote)
                self._cached_py_requires[resolved_ref] = py_require
            result.add_pyrequire(py_require)
        return result

    def _resolve_ref(self, requirement, graph_lock, remotes, update):
        if requirement.alias:
            raise ConanException("python-requires 'alias' are not supported in Conan 2.0. "
                                 "Please use version ranges instead")
        if graph_lock:
            graph_lock.resolve_locked_pyrequires(requirement, self._resolve_prereleases)
        # If the lock hasn't resolved the range, and it hasn't failed (it is partial), resolve it
        self._range_resolver.resolve(requirement, "python_requires", remotes, update)
        ref = requirement.ref
        return ref

    def _load_pyreq_conanfile(self, loader, graph_lock, ref, remotes, update, check_update):
        try:
            recipe = self._proxy.get_recipe(ref, remotes, update, check_update)
        except ConanException as e:
            raise ConanException(f"Cannot resolve python_requires '{ref}': {str(e)}")
        layout, recipe_status, remote = recipe
        path = layout.conanfile()
        new_ref = layout.reference
        conanfile, module = loader.load_basic_module(path, graph_lock, remotes=remotes,
                                                     update=update, check_update=check_update)
        conanfile.name = new_ref.name
        conanfile.version = str(new_ref.version)
        conanfile.user = new_ref.user
        # TODO: Is this really necessary?
        conanfile.channel = new_ref.channel

        if getattr(conanfile, "alias", None):
            raise ConanException("python-requires 'alias' are not supported in Conan 2.0. "
                                 "Please use version ranges instead")
        return conanfile, module, new_ref, os.path.dirname(path), recipe_status, remote
