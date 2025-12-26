import copy
import fnmatch
import json
import os
from os.path import join, normpath

from conan.api.model import RecipeReference
from conan.internal.util.files import load, save


EDITABLE_PACKAGES_FILE = 'editable_packages.json'


class EditablePackages:
    def __init__(self, cache_folder=None):
        if cache_folder is None:
            self._edited_refs = {}
            return
        self._edited_file = normpath(join(cache_folder, EDITABLE_PACKAGES_FILE))
        if os.path.exists(self._edited_file):
            edited = load(self._edited_file)
            edited_js = json.loads(edited)
            self._edited_refs = {RecipeReference.loads(r): d
                                 for r, d in edited_js.items()}
        else:
            self._edited_refs = {}  # {ref: {"path": path, "layout": layout}}

    def update_copy(self, ws_editables):
        """
        Create a new instance with the union of the editable packages of self and other
        """
        if ws_editables is None:
            return self
        result = EditablePackages()
        result._edited_refs = self._edited_refs.copy()
        result._edited_refs.update(ws_editables)
        return result

    @property
    def edited_refs(self):
        return self._edited_refs

    def save(self):
        d = {str(ref): d for ref, d in self._edited_refs.items()}
        save(self._edited_file, json.dumps(d))

    def get(self, ref):
        _tmp = copy.copy(ref)
        _tmp.revision = None
        return self._edited_refs.get(_tmp)

    def get_path(self, ref):
        editable = self.get(ref)
        if editable is not None:
            return editable["path"]

    def add(self, ref, path, output_folder=None):
        assert isinstance(ref, RecipeReference)
        _tmp = copy.copy(ref)
        _tmp.revision = None
        self._edited_refs[ref] = {"path": path, "output_folder": output_folder}
        self.save()

    def remove(self, path, requires):
        removed = {}
        kept = {}
        for ref, info in self._edited_refs.items():
            to_remove = False
            if path and info["path"] == path:
                to_remove = True
            else:
                for r in requires or []:
                    if fnmatch.fnmatch(str(ref), r):
                        to_remove = True
            if to_remove:
                removed[ref] = info
            else:
                kept[ref] = info
        self._edited_refs = kept
        self.save()
        return removed
