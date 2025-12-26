import inspect
import os
import traceback
import importlib

from conan.internal.cache.home_paths import HomePaths
from conan.internal.subsystems import deduce_subsystem, subsystem_path
from conan.internal.errors import conanfile_exception_formatter
from conan.errors import ConanException
from conan.internal.util.files import save, mkdir, chdir

_generators = {"CMakeToolchain": "conan.tools.cmake",
               "CMakeDeps": "conan.tools.cmake",
               "CMakeConfigDeps": "conan.tools.cmake",
               "MesonToolchain": "conan.tools.meson",
               "MSBuildDeps": "conan.tools.microsoft",
               "MSBuildToolchain": "conan.tools.microsoft",
               "NMakeToolchain": "conan.tools.microsoft",
               "NMakeDeps": "conan.tools.microsoft",
               "VCVars": "conan.tools.microsoft",
               "VirtualRunEnv": "conan.tools.env.virtualrunenv",
               "VirtualBuildEnv": "conan.tools.env.virtualbuildenv",
               "AutotoolsDeps": "conan.tools.gnu",
               "AutotoolsToolchain": "conan.tools.gnu",
               "GnuToolchain": "conan.tools.gnu",
               "PkgConfigDeps": "conan.tools.gnu",
               "BazelDeps": "conan.tools.google",
               "BazelToolchain": "conan.tools.google",
               "IntelCC": "conan.tools.intel",
               "XcodeDeps": "conan.tools.apple",
               "XcodeToolchain": "conan.tools.apple",
               "PremakeDeps": "conan.tools.premake",
               "PremakeToolchain": "conan.tools.premake",
               "MakeDeps": "conan.tools.gnu",
               "SConsDeps": "conan.tools.scons",
               "QbsDeps": "conan.tools.qbs",
               "QbsProfile": "conan.tools.qbs",
               "CPSDeps": "conan.tools.cps",
               "ROSEnv": "conan.tools.ros"
               }


def _get_generator_class(generator_name):
    try:
        generator_class = _generators[generator_name]
        # This is identical to import ... form ... in terms of cacheing
    except KeyError as e:
        raise ConanException(f"Invalid generator '{generator_name}'. "
                             f"Available types: {', '.join(_generators)}") from e
    try:
        return getattr(importlib.import_module(generator_class), generator_name)
    except ImportError as e:
        raise ConanException("Internal Conan error: "
                             f"Could not find module {generator_class}") from e
    except AttributeError as e:
        raise ConanException("Internal Conan error: "
                             f"Could not find name {generator_name} "
                             f"inside module {generator_class}") from e


def load_cache_generators(path):
    from conan.internal.loader import load_python_file
    result = {}  # Name of the generator: Class
    if not os.path.isdir(path):
        return result
    for f in os.listdir(path):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        full_path = os.path.join(path, f)
        mod, _ = load_python_file(full_path)
        for name, value in inspect.getmembers(mod):
            if inspect.isclass(value) and not name.startswith("_"):
                result[name] = value
    return result


def write_generators(conanfile, hook_manager, home_folder, envs_generation=None):
    new_gen_folder = conanfile.generators_folder
    _receive_conf(conanfile)
    _receive_generators(conanfile)

    # TODO: Optimize this, so the global generators are not loaded every call to write_generators
    global_generators = load_cache_generators(HomePaths(home_folder).custom_generators_path)
    hook_manager.execute("pre_generate", conanfile=conanfile)

    if conanfile.generators:
        conanfile.output.highlight(f"Writing generators to {new_gen_folder}")
    # generators check that they are not present in the generators field,
    # to avoid duplicates between the generators attribute and the generate() method
    # They would raise an exception here if we don't invalidate the field while we call them
    old_generators = []
    for gen in conanfile.generators:
        if gen not in old_generators:
            old_generators.append(gen)
    conanfile.generators = []
    try:
        for generator_name in old_generators:
            if isinstance(generator_name, str):
                global_generator = global_generators.get(generator_name)
                generator_class = global_generator or _get_generator_class(generator_name)
            else:
                generator_class = generator_name
                generator_name = generator_class.__name__
            if generator_class:
                try:
                    generator = generator_class(conanfile)
                    mkdir(new_gen_folder)
                    conanfile.output.info(f"Generator '{generator_name}' calling 'generate()'")
                    with chdir(new_gen_folder):
                        generator.generate()
                    continue
                except Exception as e:
                    # When a generator fails, it is very useful to have the whole stacktrace
                    if not isinstance(e, ConanException):
                        conanfile.output.error(traceback.format_exc(), error_type="exception")
                    raise ConanException(f"Error in generator '{generator_name}': {str(e)}") from e
    finally:
        # restore the generators attribute, so it can raise
        # if the user tries to instantiate a generator already present in generators
        conanfile.generators = old_generators
    if hasattr(conanfile, "generate"):
        conanfile.output.highlight("Calling generate()")
        conanfile.output.info(f"Generators folder: {new_gen_folder}")
        mkdir(new_gen_folder)
        with chdir(new_gen_folder):
            with conanfile_exception_formatter(conanfile, "generate"):
                conanfile.generate()

    if envs_generation is None:
        if conanfile.virtualbuildenv:
            mkdir(new_gen_folder)
            with chdir(new_gen_folder):
                from conan.tools.env.virtualbuildenv import VirtualBuildEnv
                env = VirtualBuildEnv(conanfile)
                # TODO: Check length of env.vars().keys() when adding NotEmpty
                env.generate()
        if conanfile.virtualrunenv:
            mkdir(new_gen_folder)
            with chdir(new_gen_folder):
                from conan.tools.env import VirtualRunEnv
                env = VirtualRunEnv(conanfile)
                env.generate()

    _generate_aggregated_env(conanfile)

    hook_manager.execute("post_generate", conanfile=conanfile)


def _receive_conf(conanfile):
    """  collect conf_info from the immediate build_requires, aggregate it and injects/update
    current conf
    """
    # TODO: Open question 1: Only build_requires can define config?
    # TODO: Only direct build_requires?
    # TODO: Is really the best mechanism to define this info? Better than env-vars?
    # Conf only for first level build_requires
    for build_require in conanfile.dependencies.direct_build.values():
        if build_require.conf_info:
            conanfile.conf.compose_conf(build_require.conf_info)


def _receive_generators(conanfile):
    """  Collect generators_info from the immediate build_requires"""
    for build_req in conanfile.dependencies.direct_build.values():
        if build_req.generator_info:
            if not isinstance(build_req.generator_info, list):
                raise ConanException(f"{build_req} 'generator_info' must be a list")
            names = [c.__name__ if not isinstance(c, str) else c for c in build_req.generator_info]
            conanfile.output.warning(f"Tool-require {build_req} adding generators: {names}",
                                     warn_tag="experimental")
            # Generators can be defined as a tuple in recipes, ensure we don't break if so
            conanfile.generators = build_req.generator_info + list(conanfile.generators)


def _generate_aggregated_env(conanfile):

    def deactivates(filenames):
        # FIXME: Probably the order needs to be reversed
        result = []
        for s in reversed(filenames):
            folder, f = os.path.split(s)
            result.append(os.path.join(folder, "deactivate_{}".format(f)))
        return result

    def deactivate_function_names(filenames):
        return [os.path.splitext(os.path.basename(s))[0].replace("-", "_")
                for s in reversed(filenames)]

    deactivation_mode = conanfile.conf.get("tools.env:deactivation_mode", default=None, check_type=str)
    generated = []
    for group, env_scripts in conanfile.env_scripts.items():
        subsystem = deduce_subsystem(conanfile, group)
        bats = []
        shs = []
        ps1s = []
        for env_script in env_scripts:
            path = os.path.join(conanfile.generators_folder, env_script)
            # Only the .bat and .ps1 are made relative to current script
            if env_script.endswith(".bat"):
                path = os.path.relpath(path, conanfile.generators_folder)
                bats.append("%~dp0/"+path)
            elif env_script.endswith(".sh"):
                shs.append(subsystem_path(subsystem, path))
            elif env_script.endswith(".ps1"):
                path = os.path.relpath(path, conanfile.generators_folder)
                # This $PSScriptRoot uses the current script directory
                ps1s.append("$PSScriptRoot/"+path)
        if shs:
            def sh_content(files):
                content = ". " + " && . ".join('"{}"'.format(s) for s in files)
                if deactivation_mode == "function":
                    content += f"\n\ndeactivate_conan{group}() {{\n"
                    for deactivate_name in deactivate_function_names(shs):
                        content += f"    deactivate_{deactivate_name}\n"
                    content += f"    unset -f deactivate_conan{group}\n}}\n"
                return content
            filename = "conan{}.sh".format(group)
            generated.append(filename)
            save(os.path.join(conanfile.generators_folder, filename), sh_content(shs))
            if not deactivation_mode:
                save(os.path.join(conanfile.generators_folder, "deactivate_{}".format(filename)),
                     sh_content(deactivates(shs)))
        if bats:
            def bat_content(files):
                return "\r\n".join(["@echo off"] + ['call "{}"'.format(b) for b in files])
            filename = "conan{}.bat".format(group)
            generated.append(filename)
            save(os.path.join(conanfile.generators_folder, filename), bat_content(bats))
            save(os.path.join(conanfile.generators_folder, "deactivate_{}".format(filename)),
                 bat_content(deactivates(bats)))
        if ps1s:
            def ps1_content(files):
                content = "\r\n".join(['& "{}"'.format(b) for b in files])
                if deactivation_mode == "function":
                    content += f"\n\nfunction global:deactivate_conan{group} {{\n"
                    for deactivate_name in deactivate_function_names(ps1s):
                        content += f"    deactivate_{deactivate_name}\n"
                    content += (f"    Remove-Item -Path function:deactivate_conan{group} "
                                "-ErrorAction SilentlyContinue"
                                "\n}\n")
                return content
            filename = "conan{}.ps1".format(group)
            generated.append(filename)
            save(os.path.join(conanfile.generators_folder, filename), ps1_content(ps1s))
            if not deactivation_mode:
                save(os.path.join(conanfile.generators_folder, "deactivate_{}".format(filename)),
                     ps1_content(deactivates(ps1s)))
    if generated:
        conanfile.output.highlight("Generating aggregated env files")
        conanfile.output.info(f"Generated aggregated env files: {generated}")


def relativize_paths(conanfile, placeholder):
    abs_base_path = conanfile.folders._base_generators
    if not abs_base_path or not os.path.isabs(abs_base_path):
        return None, None
    abs_base_path = os.path.join(abs_base_path, "")  # For the trailing / to dissambiguate matches
    generators_folder = conanfile.generators_folder
    try:
        rel_path = os.path.relpath(abs_base_path, generators_folder)
    except ValueError:  # In case the unit in Windows is different, path cannot be made relative
        return None, None
    new_path = placeholder if rel_path == "." else os.path.join(placeholder, rel_path)
    new_path = os.path.join(new_path, "")  # For the trailing / to dissambiguate matches
    return abs_base_path, new_path


def relativize_path(path, conanfile, placeholder, normalize=True):
    """
    relative path from the "generators_folder" to "path", asuming the root file, like
    conan_toolchain.cmake will be directly in the "generators_folder"
    """
    base_common_folder = conanfile.folders._base_generators # noqa
    if not base_common_folder or not os.path.isabs(base_common_folder):
        return path
    try:
        common_path = os.path.commonpath([path, conanfile.generators_folder, base_common_folder])
        if common_path.replace("\\", "/") == base_common_folder.replace("\\", "/"):
            rel_path = os.path.relpath(path, conanfile.generators_folder)
            new_path = os.path.join(placeholder, rel_path)
            return new_path.replace("\\", "/") if normalize else new_path
    except ValueError:  # In case the unit in Windows is different, path cannot be made relative
        pass
    return path
