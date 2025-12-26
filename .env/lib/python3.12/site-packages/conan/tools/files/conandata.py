import os

import yaml

from conan.errors import ConanException
from conan.internal.util.files import load, save


def update_conandata(conanfile, data):
    """
    Tool to modify the ``conandata.yml`` once it is exported. It can be used, for example:

       - To add additional data like the "commit" and "url" for the scm.
       - To modify the contents cleaning the data that belong to other versions (different
         from the exported) to avoid changing the recipe revision when the changed data doesn't
         belong to the current version.

    :param conanfile: The current recipe object. Always use ``self``.
    :param data: (Required) A dictionary (can be nested), of values to update
    """

    if not hasattr(conanfile, "export_folder") or conanfile.export_folder is None:
        raise ConanException("The 'update_conandata()' can only be used in the 'export()' method")
    path = os.path.join(conanfile.export_folder, "conandata.yml")
    if os.path.exists(path):
        conandata = load(path)
        conandata = yaml.safe_load(conandata)
    else:  # File doesn't exist, create it
        conandata = {}

    def recursive_dict_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    recursive_dict_update(conandata, data)
    new_content = yaml.safe_dump(conandata)
    save(path, new_content)


def trim_conandata(conanfile, raise_if_missing=True):
    """
    Tool to modify the ``conandata.yml`` once it is exported, to limit it to the current version
    only
    """
    if not hasattr(conanfile, "export_folder") or conanfile.export_folder is None:
        raise ConanException("The 'trim_conandata()' tool can only be used in "
                             "the 'export()' method or 'post_export()' hook")
    path = os.path.join(conanfile.export_folder, "conandata.yml")
    if not os.path.exists(path):
        if raise_if_missing:
            raise ConanException("conandata.yml file doesn't exist")
        else:
            conanfile.output.warning("conandata.yml file doesn't exist")
            return

    conandata = load(path)
    conandata = yaml.safe_load(conandata)

    version = str(conanfile.version)
    result = {}
    for k, v in conandata.items():
        if k == "scm" or not isinstance(v, dict):
            result[k] = v
            continue  # to allow user extra conandata, common to all versions
        version_data = v.get(version)
        if version_data is not None:
            result[k] = {version: version_data}

    # Update the internal conanfile data too
    conanfile.conan_data = result

    new_conandata_yml = yaml.safe_dump(result, default_flow_style=False)
    save(path, new_conandata_yml)
