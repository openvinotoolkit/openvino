import os

from conan.api.model.list import PackagesList
from conan.api.output import ConanOutput
from conan.api.model import PkgReference
from conan.api.model import RecipeReference


class IntegrityChecker:
    """
    Check:
        - Performs a corruption integrity check in the cache. This is done by loading the existing
        conanmanifest.txt and comparing against a computed conanmanifest.txt. It
        doesn't address someone tampering with the conanmanifest.txt, just accidental
        modifying of a package contents, like if some file has been added after computing the
        manifest.
        This is to be done over the package contents, not the compressed conan_package.tgz
        artifacts
    """
    def __init__(self, cache):
        self._cache = cache

    def check(self, pkg_list) -> PackagesList:
        corrupted_pkglist = PackagesList()
        for ref, packages in pkg_list.items():
            # Check if any of the packages are corrupted
            if self._recipe_corrupted(ref):
                # If the recipe is corrupted, all its packages are considered corrupted
                corrupted_pkglist.add_ref(ref)
            else:
                # Do not check any binary if the recipe is corrupted
                for pref in packages:
                    if self._package_corrupted(pref):
                        corrupted_pkglist.add_ref(ref)
                        # Cannot add package reference without having the recipe reference already added
                        corrupted_pkglist.add_pref(pref)
        return corrupted_pkglist

    def _recipe_corrupted(self, ref: RecipeReference):
        layout = self._cache.recipe_layout(ref)
        output = ConanOutput(scope=f"{ref.repr_notime()}")
        try:
            read_manifest, expected_manifest = layout.recipe_manifests()
        except FileNotFoundError:
            output.error("Manifest missing", error_type="exception")
            return True
        # Filter exports_sources from read manifest if there are no exports_sources locally
        # This happens when recipe is downloaded without sources (not built from source)
        export_sources_folder = layout.export_sources()
        if not os.path.exists(export_sources_folder):
            read_manifest.file_sums = {k: v for k, v in read_manifest.file_sums.items()
                                       if not k.startswith("export_source")}

        if read_manifest != expected_manifest:
            output_lines = ["", "Manifest mismatch", f"    Folder: {layout.package()}"]
            diff = read_manifest.difference(expected_manifest)
            for fname, (h1, h2) in diff.items():
                output_lines.append(f"        {fname} (manifest: {h1}, file: {h2})")
            output.error("\n".join(output_lines), error_type="exception")
            return True
        output.info("Integrity check: ok")

    def _package_corrupted(self, ref: PkgReference):
        layout = self._cache.pkg_layout(ref)
        output = ConanOutput(scope=f"{ref.repr_notime()}")
        try:
            read_manifest, expected_manifest = layout.package_manifests()
        except FileNotFoundError:
            output.error("Manifest missing", error_type="exception")
            return True

        if read_manifest != expected_manifest:
            output_lines = ["", "Manifest mismatch", f"    Folder: {layout.package()}"]
            diff = read_manifest.difference(expected_manifest)
            for fname, (h1, h2) in diff.items():
                output_lines.append(f"        {fname} (manifest: {h1}, file: {h2})")
            output.error("\n".join(output_lines), error_type="exception")
            return True
        output.info("Integrity check: ok")
