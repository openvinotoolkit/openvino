import os
import shutil

import yaml

from conan.api.output import ConanOutput
from conan.errors import ConanException
from conan.internal.errors import scoped_traceback
from conan.internal.util.files import load, save

# Related folder
WORKSPACE_FOLDER = "conanws"
# Related files
WORKSPACE_YML = "conanws.yml"
WORKSPACE_PY = "conanws.py"


class Workspace:
    """
    The base class for all workspaces
    """

    def __init__(self, folder, conan_api):
        self.folder = folder
        self.conan_data = self._conan_load_data()
        self._conan_api = conan_api
        self.output = ConanOutput(scope=f"Workspace '{self.name()}'")

    def __getattribute__(self, item):
        # Return a protected wrapper around workspace overridable callables in order to
        # be able to have clean errors if user errors in conanws.py code
        myattr = object.__getattribute__(self, item)
        if item not in ("name", "packages", "add", "remove", "clean", "build_order"):
            return myattr

        def wrapper(*args, **kwargs):
            try:
                return myattr(*args, **kwargs)
            except ConanException:
                raise
            except Exception as e:
                m = scoped_traceback(f"Error in {item}() method", e, scope="conanws.py")
                raise ConanException(f"Workspace conanws.py file: {m}")
        return wrapper

    def name(self):
        return self.conan_data.get("name") or os.path.basename(self.folder)

    def _conan_load_data(self):
        data_path = os.path.join(self.folder, WORKSPACE_YML)
        if not os.path.exists(data_path):
            return {}
        try:
            data = yaml.safe_load(load(data_path))
        except Exception as e:
            raise ConanException("Invalid yml format at {}: {}".format(WORKSPACE_YML, e))
        return data or {}

    def add(self, ref, path, output_folder):
        if not path or not os.path.isfile(path):
            raise ConanException(f"Cannot add to workspace. File not found: {path}")
        path = self._conan_rel_path(os.path.dirname(path))
        editable = {
            "path": path,
            "ref": str(ref)
        }
        if output_folder:
            editable["output_folder"] = self._conan_rel_path(output_folder)
        packages = self.conan_data.setdefault("packages", [])
        for p in packages:
            if p["path"] == path:
                self.output.warning(f"Package {path} already exists, updating its reference")
                p["ref"] = editable["ref"]
                break
        else:
            packages.append(editable)
        save(os.path.join(self.folder, WORKSPACE_YML), yaml.dump(self.conan_data))

    def remove(self, path):
        path = self._conan_rel_path(path)
        package_found = next((package_info for package_info in self.conan_data.get("packages", [])
                              if package_info["path"].replace("\\", "/") == path), None)
        if not package_found:
            raise ConanException(f"No editable package to remove from this path: {path}")
        self.conan_data["packages"].remove(package_found)
        save(os.path.join(self.folder, WORKSPACE_YML), yaml.dump(self.conan_data))
        return path

    def clean(self):
        self.output.info("Default workspace clean: Removing the output-folder of each editable")
        for package_info in self.conan_data.get("packages", []):
            editable_label = package_info.get("ref", "") or package_info['path']
            if not package_info.get("output_folder"):
                self.output.info(f"Editable {editable_label} doesn't have an output_folder defined")
                continue
            of = os.path.join(self.folder, package_info["output_folder"])
            try:
                self.output.info(f"Removing {editable_label} output folder: {of}")
                shutil.rmtree(of)
            except OSError as e:
                self.output.warning(f"Error removing {editable_label} output folder: {str(e)}")

    def _conan_rel_path(self, path):
        if path is None:
            return None
        if not os.path.isabs(path):
            raise ConanException(f"Editable path must be absolute: {path}")
        path = os.path.relpath(path, self.folder)
        return path.replace("\\", "/")  # Normalize to unix path

    def packages(self):
        return self.conan_data.get("packages", [])

    def load_conanfile(self, conanfile_path):
        conanfile_path = os.path.join(self.folder, conanfile_path, "conanfile.py")
        from conan.internal.loader import ConanFileLoader
        from conan.internal.cache.home_paths import HomePaths
        from conan.internal.conan_app import ConanFileHelpers, CmdWrapper
        cmd_wrap = CmdWrapper(HomePaths(self._conan_api.home_folder).wrapper_path)
        helpers = ConanFileHelpers(None, cmd_wrap, self._conan_api._api_helpers.global_conf,
                                   cache=None, home_folder=self._conan_api.home_folder)
        loader = ConanFileLoader(pyreq_loader=None, conanfile_helpers=helpers)
        conanfile = loader.load_named(conanfile_path, name=None, version=None, user=None,
                                      channel=None, remotes=None, graph_lock=None)
        return conanfile

    def root_conanfile(self):  # noqa
        return None

    def build_order(self, order):  # noqa
        msg = ["Packages build order:"]
        for level in order:
            for item in level:
                msg.append(f"    {item['ref']}: {item['folder']}")
        self.output.info("\n".join(msg))
