from conan.tools.files import save
from conan.errors import ConanException
from conan.internal.model.dependencies import get_transitive_requires
import json
import os


class _QbsDepsModuleFile:
    def __init__(self, qbsdeps, dep, component, deps, module_name):
        self._qbsdeps = qbsdeps
        self._dep = dep
        self._component = component
        self._deps = deps
        self._module_name = module_name
        self._build_bindirs = qbsdeps._build_bindirs
        self._version = (component.get_property("component_version") or
                         component.get_property("system_package_version") or
                         dep.ref.version)

    @property
    def filename(self):
        return self._module_name + '.json'

    @property
    def version(self):
        return self._version

    def get_content(self):
        cpp_info_attrs = [
            'includedirs', 'srcdirs', 'libdirs', 'resdirs', 'bindirs', 'builddirs',
            'frameworkdirs', 'system_libs', 'frameworks', 'libs', 'defines', 'cflags', 'cxxflags',
            'sharedlinkflags', 'exelinkflags', 'objects', 'sysroot'
        ]
        return {
            'package_name': self._dep.ref.name,
            'package_dir': self._get_package_dir(),
            'version': str(self._version),
            'cpp_info': {k : getattr(self._component, k) for k in cpp_info_attrs},
            'build_bindirs': self._build_bindirs,
            'dependencies': [{'name': n, "version": str(v)} for n, v in self._deps],
            'settings': {k: v for k, v in self._dep.settings.items()},
            'options': {k: v for k, v in self._dep.options.items()}
        }

    def _get_package_dir(self):
        # If editable, package_folder can be None
        root_folder = self._dep.recipe_folder if self._dep.package_folder is None \
            else self._dep.package_folder
        return root_folder.replace("\\", "/")

    def render(self):
        return json.dumps(self.get_content(), indent=4)


class _QbsDepGenerator:
    """ Handles a single package, can create multiple modules in case of several components
    """
    def __init__(self, conanfile, dep, build_bindirs):
        self._conanfile = conanfile
        self._dep = dep
        self._build_bindirs = build_bindirs

    @property
    def content(self):
        qbs_files = {}
        transitive_reqs = get_transitive_requires(self._conanfile, self._dep)

        def _get_package_name(dep):
            # TODO: pkgconfig uses suffix, do we need it? see:
            # https://github.com/conan-io/conan/blob/develop2/conan/tools/gnu/pkgconfigdeps.py#L319
            return dep.cpp_info.get_property("pkg_config_name") or dep.ref.name

        def _get_component_name(dep, comp_name):
            if comp_name not in dep.cpp_info.components:
                if dep.ref.name == comp_name:
                    return _get_package_name(dep)
                raise ConanException("Component '{name}::{cname}' not found in '{name}' "
                                     "package requirement".format(name=dep.ref.name,
                                                                  cname=comp_name))

            # TODO: pkgconfig uses suffix, do we need it?
            # We re-use pkg_config_name for compatiblitity with the Qbs pkg-config provider:
            # in that case, Qbs/its users do not need to do additional mapping on their side
            pkg_config_name = dep.cpp_info.components[comp_name].get_property("pkg_config_name")
            return pkg_config_name or comp_name

        def _get_name_with_namespace(namespace, name):
            """
            Build a name with a namespace, e.g., openssl-crypto
            """
            return f"{namespace}-{name}"

        def get_components(dep):
            ret = {}
            for comp_ref_name, info in dep.cpp_info.get_sorted_components().items():
                comp_name = _get_component_name(dep, comp_ref_name)
                ret[comp_name] = info
            return ret

        # copy & paste from pkgconfig deps
        def get_cpp_info_requires_names(dep, cpp_info):
            ret = []
            dep_ref_name = dep.ref.name
            for req in cpp_info.requires:
                pkg_ref_name, comp_ref_name = (
                    req.split("::") if "::" in req else (dep_ref_name, req)
                )

                if dep_ref_name != pkg_ref_name:
                    try:
                        req_conanfile = transitive_reqs[pkg_ref_name]
                    except KeyError:
                        # If the dependency is not in the transitive, might be skipped
                        continue
                # For instance, dep == "hello/1.0" and req == "hello::cmp1" -> hello == hello
                else:
                    req_conanfile = dep

                comp_name = _get_component_name(req_conanfile, comp_ref_name)
                if not comp_name:
                    pkg_name = _get_package_name(req_conanfile)
                    # Creating a component name with namespace, e.g., dep-comp1
                    comp_name = _get_name_with_namespace(pkg_name, comp_ref_name)
                ret.append((comp_name, req_conanfile.ref.version))
            return ret

        if not self._dep.cpp_info.has_components:
            module_name = _get_package_name(self._dep)
            requires = get_cpp_info_requires_names(self._dep, self._dep.cpp_info)
            if not requires:
                # If no requires were found, let's try to get all the direct visible
                # dependencies, e.g., requires = "other_pkg/1.0"
                for deprequire, _ in self._dep.dependencies.direct_host.items():
                    requires.append((deprequire.ref.name, deprequire.ref.version))
            file = _QbsDepsModuleFile(
                self, self._dep, self._dep.cpp_info, requires, module_name
            )
            qbs_files[file.filename] = file
        else:
            full_requires = []
            for module_name, component in get_components(self._dep).items():
                requires = get_cpp_info_requires_names(self._dep, component)
                file = _QbsDepsModuleFile(self, self._dep, component, requires, module_name)
                qbs_files[file.filename] = file
                full_requires.append((module_name, file.version))
            module_name = _get_package_name(self._dep)
            file = _QbsDepsModuleFile(
                self, self._dep, self._dep.cpp_info, full_requires, module_name)
            # We create the root package's module file ONLY
            # if it does not already exist in components
            # An example is a grpc package where they have a "grpc" component
            if file.filename not in qbs_files:
                qbs_files[file.filename] = file

        return qbs_files


class QbsDeps:
    """
    This class will generate a JSON file for each dependency inside the "conan-qbs-deps" folder.
    Each JSON file contains information necesary for Qbs ``"conan" module provider`` to be
    able to generate Qbs module files.
    """
    def __init__(self, conanfile):
        """
        :param conanfile: The current recipe object. Always use ``self``.
        """
        self._conanfile = conanfile

    @property
    def content(self):
        """
        Returns all dependency information as a Python dict object where key is the dependency
        name and value is a dict with dependency properties.
        """
        qbs_files = {}

        build_bindirs = {
            dep.ref.name: dep.cpp_info.bindirs
            for _, dep in self._conanfile.dependencies.build.items()}

        for require, dep in self._conanfile.dependencies.items():

            # skip build deps for now
            if require.build:
                continue

            dep_build_bindirs = build_bindirs.get(dep.ref.name, [])
            qbs_files.update(_QbsDepGenerator(self._conanfile, dep, dep_build_bindirs).content)
        return qbs_files

    def generate(self):
        """
        This method will save the generated files to the "conan-qbs-deps" directory inside the
        ``conanfile.generators_folder`` directory.
        Generates a single JSON file per dependency or component.
        """
        for file_name, qbs_deps_file in self.content.items():
            save(self._conanfile, os.path.join('conan-qbs-deps', file_name), qbs_deps_file.render())
