import filecmp
import os
import shutil
import fnmatch

from conan.internal.cache.home_paths import HomePaths
from conan.api.output import ConanOutput
from conan.internal.loader import load_python_file
from conan.internal.errors import conanfile_exception_formatter
from conan.errors import ConanException
from conan.internal.util.files import rmdir, mkdir


def _find_deployer(d, cache_deploy_folder):
    """ Implements the logic of finding a deployer, with priority:
    - 1) absolute paths
    - 2) relative to cwd
    - 3) in the cache/extensions/deploy folder
    - 4) built-in
    """
    def _load(path):
        mod, _ = load_python_file(path)
        try:
            return mod.deploy
        except AttributeError:
            raise ConanException(f"Deployer does not contain 'deploy()' function: {path}")

    if not d.endswith(".py"):
        d += ".py"  # Deployers must be python files
    if os.path.isabs(d):
        return _load(d)
    cwd = os.getcwd()
    local_path = os.path.normpath(os.path.join(cwd, d))
    if os.path.isfile(local_path):
        return _load(local_path)
    cache_path = os.path.join(cache_deploy_folder, d)
    if os.path.isfile(cache_path):
        return _load(cache_path)
    builtin_deploy = {"full_deploy.py": full_deploy,
                      "direct_deploy.py": direct_deploy,
                      "runtime_deploy.py": runtime_deploy}.get(d)
    if builtin_deploy is not None:
        return builtin_deploy
    raise ConanException(f"Cannot find deployer '{d}'")


def do_deploys(home_folder, graph, deploy, deploy_package, deploy_folder):
    try:
        mkdir(deploy_folder)
    except Exception as e:
        raise ConanException(f"Deployer folder cannot be created '{deploy_folder}':\n{e}")
    # handle the recipe deploy()
    if deploy_package:
        # Similar processing as BuildMode class
        excluded = [p[1:] for p in deploy_package if p[0] in ["!", "~"]]
        included = [p for p in deploy_package if p[0] not in ["!", "~"]]
        for node in graph.ordered_iterate():
            conanfile = node.conanfile
            if not conanfile.ref:  # virtual or conanfile.txt, can't have deployer
                continue
            consumer = conanfile._conan_is_consumer  # noqa
            if any(conanfile.ref.matches(p, consumer) for p in excluded):
                continue
            if not any(conanfile.ref.matches(p, consumer) for p in included):
                continue
            if hasattr(conanfile, "deploy"):
                conanfile.output.info("Executing deploy()")
                conanfile.deploy_folder = deploy_folder
                with conanfile_exception_formatter(conanfile, "deploy"):
                    conanfile.deploy()
    # Handle the deploys
    cache = HomePaths(home_folder)
    for d in deploy or []:
        deployer = _find_deployer(d, cache.deployers_path)
        # IMPORTANT: Use always kwargs to not break if it changes in the future
        deployer(graph=graph, output_folder=deploy_folder)


def full_deploy(graph, output_folder):
    """
    Deploys to output_folder + host/dep/0.1/Release/x86_64 subfolder
    """
    # TODO: This deployer needs to be put somewhere else
    # TODO: Document that this will NOT work with editables
    conanfile = graph.root.conanfile
    conanfile.output.info(f"Conan built-in full deployer to {output_folder}")
    for dep in conanfile.dependencies.values():
        if dep.package_folder is None:
            continue
        folder_name = os.path.join("full_deploy", dep.context, dep.ref.name, str(dep.ref.version))
        build_type = dep.info.settings.get_safe("build_type")
        arch = dep.info.settings.get_safe("arch")
        if build_type:
            folder_name = os.path.join(folder_name, build_type)
        if arch:
            folder_name = os.path.join(folder_name, arch)
        _deploy_single(dep, conanfile, output_folder, folder_name)


def runtime_deploy(graph, output_folder):
    """
    Deploy all the shared libraries and the executables of the dependencies in a flat directory.

    It preserves symlinks in case the configuration tools.deployer:symlinks is True.
    It preserves the directory structure when having subfolders
    """
    conanfile = graph.root.conanfile
    output = ConanOutput(scope="runtime_deploy")
    output.info(f"Deploying dependencies runtime to folder: {output_folder}")
    output.warning("This deployer is experimental and subject to change. "
                   "Please give feedback at https://github.com/conan-io/conan/issues")
    mkdir(output_folder)
    symlinks = conanfile.conf.get("tools.deployer:symlinks", check_type=bool, default=True)
    for req, dep in conanfile.dependencies.host.items():
        if not req.run:  # Avoid deploying unused binaries at runtime
            continue
        if dep.package_folder is None:
            output.warning(f"{dep.ref} does not have any package folder, skipping binary")
            continue
        count = 0
        cpp_info = dep.cpp_info.aggregated_components()
        for bindir in cpp_info.bindirs:
            if not os.path.isdir(bindir):
                output.warning(f"{dep.ref} {bindir} does not exist")
                continue
            count += _flatten_directory(dep, bindir, output_folder, symlinks)

        for libdir in cpp_info.libdirs:
            if not os.path.isdir(libdir):
                output.warning(f"{dep.ref} {libdir} does not exist")
                continue
            count += _flatten_directory(dep, libdir, output_folder, symlinks, [".dylib*", ".so*"])

        output.info(f"Copied {count} files from {dep.ref}")
    conanfile.output.success(f"Runtime deployed to folder: {output_folder}")


def _flatten_directory(dep, src_dir, output_dir, symlinks, extension_filter=None):
    """
    Copy all the files from the source directory in a flat output directory, respecting subfolders.
    An optional string, named extension_filter, can be set to copy only the files with
    the listed extensions.
    """
    file_count = 0

    output = ConanOutput(scope="runtime_deploy")
    for src_dirpath, _, src_filenames in os.walk(src_dir, followlinks=symlinks):
        rel_path = os.path.relpath(src_dirpath, src_dir)
        for src_filename in src_filenames:
            if extension_filter and not any(fnmatch.fnmatch(src_filename, f'*{ext}') for ext in extension_filter):
                continue

            src_filepath = os.path.join(src_dirpath, src_filename)
            dest_filepath = os.path.join(output_dir, rel_path, src_filename)

            if not symlinks and os.path.islink(src_filepath):
                continue

            if not os.path.exists(os.path.dirname(dest_filepath)):
                os.makedirs(os.path.dirname(dest_filepath))

            if os.path.exists(dest_filepath):
                if filecmp.cmp(src_filepath, dest_filepath):  # Be efficient, do not copy
                    output.verbose(f"{dest_filepath} exists with same contents, skipping copy")
                    continue
                else:
                    output.warning(f"{dest_filepath} exists and will be overwritten")

            try:
                file_count += 1
                # INFO: When follow_symlinks is false, and src is a symbolic link, it tries to
                # copy all metadata from the src symbolic link to the newly created dst link
                shutil.copy2(src_filepath, dest_filepath, follow_symlinks=not symlinks)
                output.verbose(f"Copied {src_filepath} into {output_dir}")
            except Exception as e:
                if "WinError 1314" in str(e):
                    ConanOutput().error("runtime_deploy: Windows symlinks require admin privileges "
                                        "or 'Developer mode = ON'", error_type="exception")
                raise ConanException(f"runtime_deploy: Copy of '{dep}' files failed: {e}.\nYou can "
                                     f"use 'tools.deployer:symlinks' conf to disable symlinks")
    return file_count


def _deploy_single(dep, conanfile, output_folder, folder_name):
    new_folder = os.path.join(output_folder, folder_name)
    rmdir(new_folder)
    symlinks = conanfile.conf.get("tools.deployer:symlinks", check_type=bool, default=True)
    try:
        shutil.copytree(dep.package_folder, new_folder, symlinks=symlinks)
    except Exception as e:
        if "WinError 1314" in str(e):
            ConanOutput().error("full_deploy: Symlinks in Windows require admin privileges "
                                "or 'Developer mode = ON'", error_type="exception")
        raise ConanException(f"full_deploy: The copy of '{dep}' files failed: {e}.\nYou can "
                             f"use 'tools.deployer:symlinks' conf to disable symlinks")
    dep.set_deploy_folder(new_folder)


def direct_deploy(graph, output_folder):
    """
    Deploys to output_folder a single package,
    """
    # TODO: This deployer needs to be put somewhere else
    # TODO: Document that this will NOT work with editables
    output_folder = os.path.join(output_folder, "direct_deploy")
    conanfile = graph.root.conanfile
    conanfile.output.info(f"Conan built-in pkg deployer to {output_folder}")
    # If the argument is --requires, the current conanfile is a virtual one with 1 single
    # dependency, the "reference" package. If the argument is a local path, then all direct
    # dependencies
    for dep in conanfile.dependencies.filter({"direct": True}).values():
        _deploy_single(dep, conanfile, output_folder, dep.ref.name)
