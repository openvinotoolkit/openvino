import hashlib
import os
import re
import shutil
import uuid
from fnmatch import translate
from typing import List

from conan.internal.cache.conan_reference_layout import RecipeLayout, PackageLayout
# TODO: Random folders are no longer accessible, how to get rid of them asap?
# TODO: We need the workflow to remove existing references.
from conan.internal.cache.db.cache_database import CacheDatabase
from conan.internal.errors import ConanReferenceAlreadyExistsInDB
from conan.errors import ConanException
from conan.api.model import PkgReference
from conan.api.model import RecipeReference
from conan.internal.util.dates import revision_timestamp_now
from conan.internal.util.files import rmdir, renamedir, mkdir


class PkgCache:
    """ Class to represent the recipes and packages storage in disk
    """

    def __init__(self, cache_folder, global_conf):
        # paths
        self._store_folder = global_conf.get("core.cache:storage_path") or \
                             os.path.join(cache_folder, "p")

        try:
            mkdir(self._store_folder)
            db_filename = os.path.join(self._store_folder, 'cache.sqlite3')
            self._base_folder = os.path.abspath(self._store_folder)
            self._db = CacheDatabase(filename=db_filename)
        except Exception as e:
            raise ConanException(f"Couldn't initialize storage in {self._store_folder}: {e}")

    @property
    def store(self):
        return self._base_folder

    @property
    def temp_folder(self):
        """ temporary folder where Conan puts exports and packages before the final revision
        is computed"""
        # TODO: Improve the path definitions, this is very hardcoded
        return os.path.join(self._base_folder, "t")

    @property
    def builds_folder(self):
        return os.path.join(self._base_folder, "b")

    def _create_path(self, relative_path, remove_contents=True):
        path = self._full_path(relative_path)
        if os.path.exists(path) and remove_contents:
            rmdir(path)
        os.makedirs(path, exist_ok=True)

    def _full_path(self, relative_path):
        # This one is used only for rmdir and mkdir operations, not returned to user
        # or stored in DB
        path = os.path.realpath(os.path.join(self._base_folder, relative_path))
        return path

    @staticmethod
    def _short_hash_path(h):
        """:param h: Unicode text to reduce"""
        h = h.encode("utf-8")
        md = hashlib.sha256()
        md.update(h)
        sha_bytes = md.hexdigest()
        # len based on: https://github.com/conan-io/conan/pull/9595#issuecomment-918976451
        # Reduce length in 3 characters 16 - 3 = 13
        return sha_bytes[0:13]

    @staticmethod
    def _get_path(ref):
        return ref.name[:5] + PkgCache._short_hash_path(ref.repr_notime())

    @staticmethod
    def _get_path_pref(pref):
        return pref.ref.name[:5] + PkgCache._short_hash_path(pref.repr_notime())

    def create_export_recipe_layout(self, ref: RecipeReference):
        """  This is a temporary layout while exporting a new recipe, because the revision is not
        computed until later. The entry is not added to DB, just a temp folder is created

        This temporary export folder will be moved to permanent when revision is computed by the
        assign_rrev() method
        """
        assert ref.revision is None, "Recipe revision should be None"
        assert ref.timestamp is None
        h = ref.name[:5] + PkgCache._short_hash_path(ref.repr_notime())
        reference_path = os.path.join("t", h)
        self._create_path(reference_path)
        return RecipeLayout(ref, os.path.join(self._base_folder, reference_path))

    def create_build_pkg_layout(self, pref: PkgReference):
        # Temporary layout to build a new package, when we don't know the package revision yet
        assert pref.ref.revision, "Recipe revision must be known to get or create the package layout"
        assert pref.package_id, "Package id must be known to get or create the package layout"
        assert pref.revision is None, "Package revision should be None"
        assert pref.timestamp is None

        random_id = str(uuid.uuid4())
        h = pref.ref.name[:5] + PkgCache._short_hash_path(pref.repr_notime() + random_id)
        package_path = os.path.join("b", h)
        self._create_path(package_path)
        return PackageLayout(pref, os.path.join(self._base_folder, package_path))

    def recipe_layout(self, ref: RecipeReference):
        """ the revision must exists, the folder must exist
        """
        assert ref.revision is not None
        ref_data = self._db.get_recipe(ref)
        ref_path = ref_data.get("path")
        ref = ref_data.get("ref")  # new revision with timestamp
        return RecipeLayout(ref, os.path.join(self._base_folder, ref_path))

    def get_latest_recipe_revision(self, ref: RecipeReference) -> RecipeReference:
        assert ref.revision is None
        ref_data = self._db.get_latest_recipe(ref)
        return ref_data.get("ref")

    def get_recipe_revisions(self, ref: RecipeReference):
        # For listing multiple revisions only
        assert ref.revision is None
        return self._db.get_recipe_revisions_references(ref)

    def pkg_layout(self, pref: PkgReference):
        """ the revision must exists, the folder must exist
        """
        assert pref.ref.revision, "Recipe revision must be known to get the package layout"
        assert pref.package_id, "Package id must be known to get the package layout"
        assert pref.revision, "Package revision must be known to get the package layout"
        pref_data = self._db.try_get_package(pref)
        pref_path = pref_data.get("path")
        # we use abspath to convert cache forward slash in Windows to backslash
        return PackageLayout(pref, os.path.abspath(os.path.join(self._base_folder, pref_path)))

    def create_ref_layout(self, ref: RecipeReference):
        """ called exclusively by:
        - RemoteManager.get_recipe()
        - cache restore
        """
        assert ref.revision, "Recipe revision must be known to create the package layout"
        reference_path = self._get_path(ref)
        self._db.create_recipe(reference_path, ref)
        self._create_path(reference_path, remove_contents=False)
        return RecipeLayout(ref, os.path.join(self._base_folder, reference_path))

    def create_pkg_layout(self, pref: PkgReference):
        """ called by:
         - RemoteManager.get_package()
         - cacje restpre
        """
        assert pref.ref.revision, "Recipe revision must be known to create the package layout"
        assert pref.package_id, "Package id must be known to create the package layout"
        assert pref.revision, "Package revision should be known to create the package layout"
        package_path = self._get_path_pref(pref)
        self._db.create_package(package_path, pref, None)
        self._create_path(package_path, remove_contents=False)
        return PackageLayout(pref, os.path.join(self._base_folder, package_path))

    def update_recipe_timestamp(self, ref: RecipeReference):
        """ when the recipe already exists in cache, but we get a new timestamp from a server
        that would affect its order in our cache """
        assert ref.revision
        assert ref.timestamp
        self._db.update_recipe_timestamp(ref)

    def search_recipes(self, pattern=None, ignorecase=True):
        # Conan references in main storage
        if pattern:
            if isinstance(pattern, RecipeReference):
                pattern = repr(pattern)
            pattern = translate(pattern)
            pattern = re.compile(pattern, re.IGNORECASE if ignorecase else 0)

        return self._db.list_references(pattern)

    def exists_prev(self, pref):
        # Used just by download to skip downloads if prev already exists in cache
        return self._db.exists_prev(pref)

    def get_latest_package_revision(self, pref: PkgReference) -> PkgReference:
        return self._db.get_latest_package_reference(pref)

    def get_package_references(self, ref: RecipeReference,
                               only_latest_prev=True) -> List[PkgReference]:
        """Get the latest package references"""
        return self._db.get_package_references(ref, only_latest_prev)

    def get_package_revisions(self, pref: PkgReference) -> List[PkgReference]:
        return self._db.get_package_revisions_references(pref)

    def get_matching_build_id(self, ref, build_id):
        return self._db.get_matching_build_id(ref, build_id)

    def remove_recipe_layout(self, layout: RecipeLayout):
        layout.remove()
        # FIXME: This is clearing package binaries from DB, but not from disk/layout
        self._db.remove_recipe(layout.reference)

    def remove_package_layout(self, layout: PackageLayout):
        layout.remove()
        self._db.remove_package(layout.reference)

    def remove_build_id(self, pref):
        self._db.remove_build_id(pref)

    def assign_prev(self, layout: PackageLayout):
        pref = layout.reference

        build_id = layout.build_id
        pref.timestamp = revision_timestamp_now()
        # Wait until it finish to really update the DB
        relpath = os.path.relpath(layout.base_folder, self._base_folder)
        relpath = relpath.replace("\\", "/")  # Uniform for Windows and Linux
        try:
            self._db.create_package(relpath, pref, build_id)
        except ConanReferenceAlreadyExistsInDB:
            # TODO: Optimize this into 1 single UPSERT operation
            # There was a previous package folder for this same package reference (and prev)
            pkg_layout = self.pkg_layout(pref)
            # We remove the old one and move the new one to the path of the previous one
            # this can be necessary in case of new metadata or build-folder because of "build_id()"
            pkg_layout.remove()
            shutil.move(layout.base_folder, pkg_layout.base_folder)  # clean unused temporary build
            layout._base_folder = pkg_layout.base_folder  # reuse existing one
            # TODO: The relpath would be the same as the previous one, it shouldn't be ncessary to
            #  update it, the update_package_timestamp() can be simplified and path dropped
            relpath = os.path.relpath(layout.base_folder, self._base_folder)
            self._db.update_package_timestamp(pref, path=relpath, build_id=build_id)

    def assign_rrev(self, layout: RecipeLayout):
        """ called at export, once the exported recipe revision has been computed, it
        can register for the first time the new RecipeReference"""
        ref = layout.reference
        assert ref.revision is not None, "Revision must exist after export"
        assert ref.timestamp is None, "Timestamp no defined yet"
        ref.timestamp = revision_timestamp_now()

        # TODO: here maybe we should block the recipe and all the packages too
        # This is the destination path for the temporary created export and export_sources folders
        # with the hash created based on the recipe revision
        new_path_relative = self._get_path(ref)

        new_path_absolute = self._full_path(new_path_relative)

        if os.path.exists(new_path_absolute):
            # If there source folder exists, export and export_sources
            # folders are already copied so we can remove the tmp ones
            rmdir(self._full_path(layout.base_folder))
        else:
            # Destination folder is empty, move all the tmp contents
            renamedir(self._full_path(layout.base_folder), new_path_absolute)

        layout._base_folder = os.path.join(self._base_folder, new_path_relative)

        # Wait until it finish to really update the DB
        try:
            self._db.create_recipe(new_path_relative, ref)
        except ConanReferenceAlreadyExistsInDB:
            # This was exported before, making it latest again, update timestamp
            ref = layout.reference
            self._db.update_recipe_timestamp(ref)

    def get_recipe_lru(self, ref):
        return self._db.get_recipe_lru(ref)

    def update_recipe_lru(self, ref):
        self._db.update_recipe_lru(ref)

    def get_package_lru(self, pref):
        return self._db.get_package_lru(pref)

    def update_package_lru(self, pref):
        self._db.update_package_lru(pref)

    def path_to_ref(self, path):
        try:
            path = os.path.relpath(path, self._base_folder)
            path = path.replace("\\", "/")  # Uniform for Windows and Linux
        except ValueError:
            raise ConanException(f"Invalid path: {path}")
        return self._db.path_to_ref(path)
