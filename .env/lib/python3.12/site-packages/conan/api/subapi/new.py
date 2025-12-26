import fnmatch
import os
import shutil

from jinja2 import Template, StrictUndefined, UndefinedError, Environment, meta

from conan.api.output import ConanOutput
from conan.errors import ConanException
from conan.internal.util.files import load, save
from conan import __version__


class NewAPI:
    _NOT_TEMPLATES = "not_templates"  # Filename containing filenames of files not to be rendered

    def __init__(self, conan_api):
        self._conan_api = conan_api

    def save_template(self, template, defines=None, output_folder=None, force=False):
        """
        Save the 'template' files in the output_folder, replacing the template variables
        with the 'defines'
        :param template: The name of the template to use
        :param defines: A list with the 'k=v' variables to replace in the template
        :param output_folder: The folder where the template files will be saved, cwd if None
        :param force: If True, overwrite the files if they already exist, otherwise raise an error
        """
        # Manually parsing the remainder
        definitions = {}
        for u in defines or []:
            try:
                k, v = u.split("=", 1)
            except ValueError:
                raise ConanException(f"Template definitions must be 'key=value', received {u}")
            k = k.replace("-", "")  # Remove possible "--name=value"
            # For variables that only show up once, no need for list to keep compatible behaviour
            if k in definitions:
                if isinstance(definitions[k], list):
                    definitions[k].append(v)
                else:
                    definitions[k] = [definitions[k], v]
            else:
                definitions[k] = v

        files = self.get_template(template)  # First priority: user folder
        is_builtin = False
        if not files:  # then, try the templates in the Conan home
            files = self.get_home_template(template)
        if files:
            template_files, non_template_files = files
        else:
            template_files = self.get_builtin_template(template)
            non_template_files = {}
            is_builtin = True

        if not template_files and not non_template_files:
            raise ConanException(f"Template doesn't exist or not a folder: {template}")

        if is_builtin and template == "workspace":  # hardcoded for the workspace special case
            definitions["name"] = "liba"
        template_files = self.render(template_files, definitions)

        # Saving the resulting files
        output = ConanOutput()
        output_folder = output_folder or os.getcwd()
        # Making sure they don't overwrite existing files
        for f, v in sorted(template_files.items()):
            path = os.path.join(output_folder, f)
            if os.path.exists(path) and not force:
                raise ConanException(f"File '{f}' already exists, and --force not defined, aborting")
            save(path, v)
            output.success("File saved: %s" % f)

        # copy non-templates
        for f, v in sorted(non_template_files.items()):
            path = os.path.join(output_folder, f)
            if os.path.exists(path) and not force:
                raise ConanException(f"File '{f}' already exists, and --force not defined, aborting")
            shutil.copy2(v, path)
            output.success("File saved: %s" % f)

    @staticmethod
    def get_builtin_template(template_name):
        from conan.internal.api.new.basic import basic_file
        from conan.internal.api.new.alias_new import alias_file
        from conan.internal.api.new.cmake_exe import cmake_exe_files
        from conan.internal.api.new.cmake_lib import cmake_lib_files
        from conan.internal.api.new.header_lib import header_only_lib_files
        from conan.internal.api.new.meson_lib import meson_lib_files
        from conan.internal.api.new.meson_exe import meson_exe_files
        from conan.internal.api.new.msbuild_lib import msbuild_lib_files
        from conan.internal.api.new.msbuild_exe import msbuild_exe_files
        from conan.internal.api.new.bazel_lib import bazel_lib_files
        from conan.internal.api.new.bazel_exe import bazel_exe_files
        from conan.internal.api.new.bazel_7_lib import bazel_lib_files_7
        from conan.internal.api.new.bazel_7_exe import bazel_exe_files_7
        from conan.internal.api.new.autotools_lib import autotools_lib_files
        from conan.internal.api.new.autoools_exe import autotools_exe_files
        from conan.internal.api.new.premake_lib import premake_lib_files
        from conan.internal.api.new.premake_exe import premake_exe_files
        from conan.internal.api.new.local_recipes_index import local_recipes_index_files
        from conan.internal.api.new.qbs_lib import qbs_lib_files
        from conan.internal.api.new.workspace import workspace_files
        new_templates = {"basic": basic_file,
                         "cmake_lib": cmake_lib_files,
                         "cmake_exe": cmake_exe_files,
                         "header_lib": header_only_lib_files,
                         "meson_lib": meson_lib_files,
                         "meson_exe": meson_exe_files,
                         "msbuild_lib": msbuild_lib_files,
                         "msbuild_exe": msbuild_exe_files,
                         # TODO: Rename xxx_7 to xxx when dropped Bazel 6.x compatibility
                         "bazel_lib": bazel_lib_files,
                         "bazel_exe": bazel_exe_files,
                         "bazel_7_lib": bazel_lib_files_7,
                         "bazel_7_exe": bazel_exe_files_7,
                         "autotools_lib": autotools_lib_files,
                         "autotools_exe": autotools_exe_files,
                         "premake_lib": premake_lib_files,
                         "premake_exe": premake_exe_files,
                         "alias": alias_file,
                         "local_recipes_index": local_recipes_index_files,
                         "qbs_lib": qbs_lib_files,
                         "workspace": workspace_files}
        template_files = new_templates.get(template_name)
        return template_files

    def get_template(self, template_folder):
        """ Load a template from a user absolute folder
        """
        if os.path.isdir(template_folder):
            return self._read_files(template_folder)

    def get_home_template(self, template_name):
        """ Load a template from the Conan home templates/command/new folder
        """
        folder_template = os.path.join(self._conan_api.home_folder, "templates", "command/new",
                                       template_name)
        if os.path.isdir(folder_template):
            return self._read_files(folder_template)

    def _read_files(self, template_folder):
        template_files, non_template_files = {}, {}
        excluded = os.path.join(template_folder, self._NOT_TEMPLATES)
        if os.path.exists(excluded):
            excluded = load(excluded)
            excluded = [] if not excluded else [s.strip() for s in excluded.splitlines() if
                                                s.strip()]
        else:
            excluded = []

        for d, _, fs in os.walk(template_folder):
            for f in fs:
                if f == self._NOT_TEMPLATES:
                    continue
                rel_d = os.path.relpath(d, template_folder) if d != template_folder else ""
                rel_f = os.path.join(rel_d, f)
                path = os.path.join(d, f)
                if not any(fnmatch.fnmatch(rel_f, exclude) for exclude in excluded):
                    template_files[rel_f] = load(path)
                else:
                    non_template_files[rel_f] = path

        return template_files, non_template_files

    @staticmethod
    def render(template_files, definitions):
        result = {}
        name = definitions.get("name", "mypkg")
        if isinstance(name, list):
            raise ConanException(f"name argument can't be multiple: {name}")
        if name != name.lower():
            raise ConanException(f"name argument must be lowercase: {name}")
        definitions["conan_version"] = __version__

        def ensure_list(key):
            value = definitions.get(key)  # Convert to list, and forget about it
            if value:
                definitions[key] = [value] if isinstance(value, str) else value

        ensure_list("requires")
        ensure_list("tool_requires")

        def as_package_name(n):
            return n.replace("-", "_").replace("+", "_")

        def as_name(ref):
            ref = as_package_name(ref)
            if '/' in ref:
                ref = ref[0:ref.index('/')]
            return ref

        definitions["package_name"] = as_package_name(name).replace(".", "_")
        definitions["as_name"] = as_name
        definitions["names"] = lambda x: ", ".join(r.split("/", 1)[0] for r in x)
        if "name" not in definitions:
            definitions["name"] = "mypkg"
        if "version" not in definitions:
            definitions["version"] = "0.1"
        version = definitions.get("version")
        if isinstance(version, list):
            raise ConanException(f"version argument can't be multiple: {version}")

        try:
            for k, v in template_files.items():
                k = Template(k, keep_trailing_newline=True, undefined=StrictUndefined).render(
                    **definitions)
                v = Template(v, keep_trailing_newline=True, undefined=StrictUndefined).render(
                    **definitions)
                if v:
                    result[k] = v
        except UndefinedError:
            template_vars = []
            for templ_str in template_files.values():
                env = Environment()
                ast = env.parse(templ_str)
                template_vars.extend(meta.find_undeclared_variables(ast))

            injected_vars = {"conan_version", "package_name", "as_name"}
            optional_vars = {"requires", "tool_requires", "output_root_dir"}
            template_vars = list(set(template_vars) - injected_vars - optional_vars)
            template_vars.sort()

            raise ConanException("Missing definitions for the template. "
                                 "Required definitions are: {}"
                                 .format(", ".join("'{}'".format(var) for var in template_vars)))
        return result
