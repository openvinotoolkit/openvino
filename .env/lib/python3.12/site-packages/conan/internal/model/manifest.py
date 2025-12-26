import os
from collections import defaultdict

from conan.internal.paths import CONAN_MANIFEST, EXPORT_SOURCES_TGZ_NAME, EXPORT_TGZ_NAME, PACKAGE_TGZ_NAME
from conan.internal.util.dates import timestamp_now, timestamp_to_str
from conan.internal.util.files import load, md5, md5sum, save, gather_files


class FileTreeManifest(object):

    def __init__(self, the_time, file_sums):
        """file_sums is a dict with filepaths and md5's: {filepath/to/file.txt: md5}"""
        self.time = the_time
        self.file_sums = file_sums

    def files(self):
        return self.file_sums.keys()

    @property
    def summary_hash(self):
        s = ["%s: %s" % (f, fmd5) for f, fmd5 in sorted(self.file_sums.items())]
        s.append("")
        return md5("\n".join(s))

    @staticmethod
    def loads(text):
        """ parses a string representation, generated with __repr__
        """
        tokens = text.split("\n")
        the_time = int(tokens[0])
        file_sums = {}
        for md5line in tokens[1:]:
            if md5line:
                filename, file_md5 = md5line.rsplit(": ", 1)
                file_sums[filename] = file_md5
        return FileTreeManifest(the_time, file_sums)

    @staticmethod
    def load(folder):
        text = load(os.path.join(folder, CONAN_MANIFEST))
        return FileTreeManifest.loads(text)

    def __repr__(self):
        # Used for serialization and saving it to disk
        ret = ["%s" % self.time]
        for file_path, file_md5 in sorted(self.file_sums.items()):
            ret.append("%s: %s" % (file_path, file_md5))
        ret.append("")
        content = "\n".join(ret)
        return content

    def __str__(self):
        """  Used for displaying the manifest in user readable format in Uploader, when the server
        manifest is newer than the cache one (and not force)
        """
        ret = ["Time: %s" % timestamp_to_str(self.time)]
        for file_path, file_md5 in sorted(self.file_sums.items()):
            ret.append("%s, MD5: %s" % (file_path, file_md5))
        ret.append("")
        content = "\n".join(ret)
        return content

    def save(self, folder, filename=CONAN_MANIFEST):
        path = os.path.join(folder, filename)
        save(path, repr(self))

    def report_summary(self, output, suffix="Copied"):
        ext_files = defaultdict(list)
        for f in self.file_sums:
            if f == "conaninfo.txt":
                continue
            _, ext = os.path.splitext(f)
            ext_files[ext].append(os.path.basename(f))
        if not ext_files:
            if suffix != "Copied":
                output.warning("No files in this package!")
            return

        for ext, files in ext_files.items():
            files_str = (": " + ", ".join(files)) if len(files) < 5 else ""
            file_or_files = "file" if len(files) == 1 else "files"
            if not ext:
                output.info("%s %d %s%s" % (suffix, len(files), file_or_files, files_str))
            else:
                output.info("%s %d '%s' %s%s" % (suffix, len(files), ext, file_or_files, files_str))

    @classmethod
    def create(cls, folder, exports_sources_folder=None):
        """ Walks a folder and create a FileTreeManifest for it, reading file contents
        from disk, and capturing current time
        """
        files, _ = gather_files(folder)
        # The folders symlinks are discarded for the manifest
        for f in (PACKAGE_TGZ_NAME, EXPORT_TGZ_NAME, CONAN_MANIFEST, EXPORT_SOURCES_TGZ_NAME):
            files.pop(f, None)

        file_dict = {}
        for name, filepath in files.items():
            # For a symlink: md5 of the pointing path, no matter if broken, relative or absolute.
            value = md5(os.readlink(filepath)) if os.path.islink(filepath) else md5sum(filepath)
            file_dict[name] = value

        if exports_sources_folder:
            export_files, _ = gather_files(exports_sources_folder)
            # The folders symlinks are discarded for the manifest
            for name, filepath in export_files.items():
                # For a symlink: md5 of the pointing path, no matter if broken, relative or absolute.
                value = md5(os.readlink(filepath)) if os.path.islink(filepath) else md5sum(filepath)
                file_dict["export_source/%s" % name] = value

        date = timestamp_now()

        return cls(date, file_dict)

    def __eq__(self, other):
        """ Two manifests are equal if file_sums
        """
        return self.file_sums == other.file_sums

    def difference(self, other):
        result = {}
        for f, h in self.file_sums.items():
            h2 = other.file_sums.get(f)
            if h != h2:
                result[f] = h, h2
        for f, h in other.file_sums.items():
            h2 = self.file_sums.get(f)
            if h != h2:
                result[f] = h2, h
        return result
