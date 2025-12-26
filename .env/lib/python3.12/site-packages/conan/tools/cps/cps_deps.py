from conan.cps.cps import CPS
from conan.tools.files import save

import json
import os


class CPSDeps:
    def __init__(self, conanfile):
        self._conanfile = conanfile

    def _config_name(self):
        build_vars = ["settings.compiler", "settings.compiler.version", "settings.arch",
                      "settings.compiler.cppstd", "settings.build_type", "options.shared"]
        ret = []
        for s in build_vars:
            group, var = s.split(".", 1)
            tmp = None
            if group == "settings":
                tmp = self._conanfile.settings.get_safe(var)
            elif group == "options":
                value = self._conanfile.options.get_safe(var)
                if value is not None:
                    if var == "shared":
                        tmp = "shared" if value else "static"
                    else:
                        tmp = "{}_{}".format(var, value)
            if tmp:
                ret.append(tmp.lower())
        return "-".join(ret)

    def generate(self):
        cps_folder = os.path.join(self._conanfile.folders.base_build, "build", "cps")
        config_name = self._config_name()
        folder = os.path.join(cps_folder, config_name, "cps")  # CMake wants a "/cps/" folder
        self._conanfile.output.info(f"[CPSDeps] folder {cps_folder}")
        deps = self._conanfile.dependencies.host.items()
        mapping = {}
        for _, dep in deps:
            self._conanfile.output.info(f"[CPSDeps]: dep {dep.ref.name}")

            cps_in_package = os.path.join(dep.package_folder, f"{dep.ref.name}.cps")
            if os.path.exists(cps_in_package):
                mapping[dep.ref.name] = cps_in_package
                continue

            cps = CPS.from_conan(dep)
            output_file = cps.save(folder)
            mapping[dep.ref.name] = output_file

        name = f"cpsmap-{config_name}.json"
        self._conanfile.output.info(f"Generating CPS mapping file: {name}")
        save(self._conanfile, os.path.join(cps_folder, name), json.dumps(mapping, indent=2))
