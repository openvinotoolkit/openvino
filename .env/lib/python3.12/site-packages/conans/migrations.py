import os

from conan import conan_version
from conan.api.output import ConanOutput
from conan.internal.loader import load_python_file
from conan.errors import ConanException, ConanMigrationError
from conan.internal.model.version import Version
from conan.internal.util.files import load, save

CONAN_VERSION = "version.txt"


class Migrator(object):

    def __init__(self, conf_path, current_version):
        self.conf_path = conf_path

        self.current_version = current_version
        self.file_version_path = os.path.join(self.conf_path, CONAN_VERSION)

    def migrate(self):
        try:
            old_version = self._load_old_version()
            if old_version is None or old_version < self.current_version:
                self._apply_migrations(old_version)
                self._update_version_file()
            elif self.current_version < old_version:  # backwards migrations
                ConanOutput().warning(f"Downgrading cache from Conan {old_version} to "
                                      f"{self.current_version}")
                self._apply_back_migrations()
                self._update_version_file()
        except Exception as e:
            ConanOutput().error(str(e), error_type="exception")
            raise ConanMigrationError(e)

    def _update_version_file(self):
        try:
            save(self.file_version_path, str(self.current_version))
        except Exception as error:
            raise ConanException("Can't write version file in '{}': {}"
                                 .format(self.file_version_path, str(error)))

    def _load_old_version(self):
        try:
            tmp = load(self.file_version_path)
            old_version = Version(tmp)
        except Exception:
            old_version = None
        return old_version

    def _apply_migrations(self, old_version):
        """
        Apply any migration script.

        :param old_version: ``str`` previous Conan version.
        """
        pass

    def _apply_back_migrations(self):
        migrations = os.path.join(self.conf_path, "migrations")
        if not os.path.exists(migrations):
            return

        # Order by versions, and filter only newer than the current version
        migration_files = []
        for f in os.listdir(migrations):
            if not f.endswith(".py"):
                continue
            version, remain = f.split("_", 1)
            version = Version(version)
            if version > conan_version:
                migration_files.append((version, remain))
        migration_files = [f"{v}_{r}" for (v, r) in reversed(sorted(migration_files))]

        for migration in migration_files:
            ConanOutput().warning(f"Applying downgrade migration {migration}")
            migration = os.path.join(migrations, migration)
            try:
                migrate_module, _ = load_python_file(migration)
                migrate_method = migrate_module.migrate
                migrate_method(self.conf_path)
            except Exception as e:
                ConanOutput().error(f"There was an error running downgrade migration: {e}. "
                                    f"Recommended to remove the cache and start from scratch",
                                    error_type="exception")
            os.remove(migration)
