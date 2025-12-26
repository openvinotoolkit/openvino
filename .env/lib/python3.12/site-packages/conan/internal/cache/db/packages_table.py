import sqlite3

from conan.internal.cache.db.table import BaseDbTable
from conan.internal.errors import ConanReferenceDoesNotExistInDB, ConanReferenceAlreadyExistsInDB
from conan.api.model import PkgReference
from conan.api.model import RecipeReference
from conan.internal.util.dates import timestamp_now


class PackagesDBTable(BaseDbTable):
    table_name = 'packages'
    columns_description = [('reference', str),
                           ('rrev', str),
                           ('pkgid', str, True),
                           ('prev', str, True),
                           ('path', str, False, True),
                           ('timestamp', float),
                           ('build_id', str, True),
                           ('lru', int)]
    unique_together = ('reference', 'rrev', 'pkgid', 'prev')

    @staticmethod
    def _as_dict(row):
        ref = RecipeReference.loads(row.reference)
        ref.revision = row.rrev
        pref = PkgReference(ref, row.pkgid, row.prev, row.timestamp)
        return {
            "pref": pref,
            "build_id": row.build_id,
            "path": row.path,
            "lru": row.lru
        }

    def _where_clause(self, pref: PkgReference):
        where_dict = {
            self.columns.reference: str(pref.ref),
            self.columns.rrev: pref.ref.revision,
            self.columns.pkgid: pref.package_id,
            self.columns.prev: pref.revision,
        }
        where_expr = ' AND '.join(
            [f"{k}='{v}' " if v is not None else f'{k} IS NULL' for k, v in where_dict.items()])
        return where_expr

    def _set_clause(self, pref: PkgReference, path=None, build_id=None):
        set_dict = {
            self.columns.reference: str(pref.ref),
            self.columns.rrev: pref.ref.revision,
            self.columns.pkgid: pref.package_id,
            self.columns.prev: pref.revision,
            self.columns.path: path,
            self.columns.timestamp: pref.timestamp,
            self.columns.build_id: build_id,
        }
        set_expr = ', '.join([f"{k} = '{v}'" for k, v in set_dict.items() if v is not None])
        return set_expr

    def get(self, pref: PkgReference):
        """ Returns the row matching the reference or fails """
        where_clause = self._where_clause(pref)
        query = f'SELECT * FROM {self.table_name} ' \
                f'WHERE {where_clause};'

        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()

        if not row:
            raise ConanReferenceDoesNotExistInDB(f"No entry for package '{repr(pref)}'")
        return self._as_dict(self.row_type(*row))

    def create(self, path, pref: PkgReference, build_id):
        assert pref.revision
        assert pref.timestamp
        # we set the timestamp to 0 until they get a complete reference, here they
        # are saved with the temporary uuid one, we don't want to consider these
        # not yet built packages for search and so on
        placeholders = ', '.join(['?' for _ in range(len(self.columns))])
        lru = timestamp_now()
        with self.db_connection() as conn:
            try:
                conn.execute(f'INSERT INTO {self.table_name} '
                             f'VALUES ({placeholders})',
                             [str(pref.ref), pref.ref.revision, pref.package_id, pref.revision,
                              path, pref.timestamp, build_id, lru])
            except sqlite3.IntegrityError:
                raise ConanReferenceAlreadyExistsInDB(f"Reference '{repr(pref)}' already exists")

    def update_timestamp(self, pref: PkgReference, path: str, build_id: str):
        assert pref.revision
        assert pref.timestamp
        where_clause = self._where_clause(pref)
        set_clause = self._set_clause(pref, path=path, build_id=build_id)
        query = f"UPDATE {self.table_name} " \
                f"SET {set_clause} " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            try:
                conn.execute(query)
            except sqlite3.IntegrityError:
                raise ConanReferenceAlreadyExistsInDB(f"Reference '{repr(pref)}' already exists")

    def update_lru(self, pref):
        assert pref.revision is not None
        # TODO: InstallGraph is dropping the pref.timestamp, cannot be checked here yet
        # assert pref.timestamp is not None, f"PREF _TIMESSTAMP IS NONE {repr(pref)}"
        where_clause = self._where_clause(pref)
        lru = timestamp_now()
        query = f"UPDATE {self.table_name} " \
                f"SET {self.columns.lru} = '{lru}' " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            conn.execute(query)

    def remove_build_id(self, pref):
        where_clause = self._where_clause(pref)
        query = f"UPDATE {self.table_name} " \
                f"SET {self.columns.build_id} = 'null' " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            try:
                conn.execute(query)
            except sqlite3.IntegrityError:
                raise ConanReferenceAlreadyExistsInDB(f"Reference '{repr(pref)}' already exists")

    def remove_recipe(self, ref: RecipeReference):
        # can't use the _where_clause, because that is an exact match on the package_id, etc
        query = f"DELETE FROM {self.table_name} " \
                f"WHERE {self.columns.reference} = '{str(ref)}' " \
                f"AND {self.columns.rrev} = '{ref.revision}' "
        with self.db_connection() as conn:
            conn.execute(query)

    def remove(self, pref: PkgReference):
        where_clause = self._where_clause(pref)
        query = f"DELETE FROM {self.table_name} " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            conn.execute(query)

    def get_package_revisions_references(self, pref: PkgReference, only_latest_prev=False):
        assert pref.ref.revision, "To search package revisions you must provide a recipe revision."
        assert pref.package_id, "To search package revisions you must provide a package id."
        check_prev = f"AND {self.columns.prev} = '{pref.revision}' " if pref.revision else ""
        if only_latest_prev:
            query = f'SELECT {self.columns.reference}, ' \
                    f'{self.columns.rrev}, ' \
                    f'{self.columns.pkgid}, ' \
                    f'{self.columns.prev}, ' \
                    f'{self.columns.path}, ' \
                    f'MAX({self.columns.timestamp}), ' \
                    f'{self.columns.build_id}, ' \
                    f'{self.columns.lru} ' \
                    f'FROM {self.table_name} ' \
                    f"WHERE {self.columns.rrev} = '{pref.ref.revision}' " \
                    f"AND {self.columns.reference} = '{str(pref.ref)}' " \
                    f"AND {self.columns.pkgid} = '{pref.package_id}' " \
                    f'{check_prev} ' \
                    f'AND {self.columns.prev} IS NOT NULL ' \
                    f'GROUP BY {self.columns.pkgid} '
        else:
            query = f'SELECT * FROM {self.table_name} ' \
                    f"WHERE {self.columns.rrev} = '{pref.ref.revision}' " \
                    f"AND {self.columns.reference} = '{str(pref.ref)}' " \
                    f"AND {self.columns.pkgid} = '{pref.package_id}' " \
                    f'{check_prev} ' \
                    f'AND {self.columns.prev} IS NOT NULL ' \
                    f'ORDER BY {self.columns.timestamp} DESC'
        with self.db_connection() as conn:
            r = conn.execute(query)
            for row in r.fetchall():
                yield self._as_dict(self.row_type(*row))

    def get_package_revisions_reference_exists(self, pref: PkgReference):
        assert pref.ref.revision, "To check package revision existence you must provide a recipe revision."
        assert pref.package_id, "To check package revisions existence you must provide a package id."
        check_prev = f"AND {self.columns.prev} = '{pref.revision}' " if pref.revision else ""
        query = f'SELECT 1 FROM {self.table_name} ' \
                f"WHERE {self.columns.rrev} = '{pref.ref.revision}' " \
                f"AND {self.columns.reference} = '{str(pref.ref)}' " \
                f"AND {self.columns.pkgid} = '{pref.package_id}' " \
                f'{check_prev} ' \
                f'AND {self.columns.prev} IS NOT NULL ' \
                'LIMIT 1 '
        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()
            return bool(row)

    def get_package_references(self, ref: RecipeReference, only_latest_prev=True):
        # Return the latest revisions
        assert ref.revision, "To search for package id's you must provide a recipe revision."
        # we select the latest prev for each package_id
        if only_latest_prev:
            query = f'SELECT {self.columns.reference}, ' \
                    f'{self.columns.rrev}, ' \
                    f'{self.columns.pkgid}, ' \
                    f'{self.columns.prev}, ' \
                    f'{self.columns.path}, ' \
                    f'MAX({self.columns.timestamp}), ' \
                    f'{self.columns.build_id}, ' \
                    f'{self.columns.lru} ' \
                    f'FROM {self.table_name} ' \
                    f"WHERE {self.columns.rrev} = '{ref.revision}' " \
                    f"AND {self.columns.reference} = '{str(ref)}' " \
                    f'GROUP BY {self.columns.pkgid} '
        else:
            query = f'SELECT * FROM {self.table_name} ' \
                    f"WHERE {self.columns.rrev} = '{ref.revision}' " \
                    f"AND {self.columns.reference} = '{str(ref)}' " \
                    f'AND {self.columns.prev} IS NOT NULL ' \
                    f'ORDER BY {self.columns.timestamp} DESC'
        with self.db_connection() as conn:
            r = conn.execute(query)
            for row in r.fetchall():
                yield self._as_dict(self.row_type(*row))

    def get_package_references_with_build_id_match(self, ref: RecipeReference, build_id):
        # Return the latest revisions
        assert ref.revision, "To search for package id's by build_id you must provide a recipe revision."
        # we select the latest prev for each package_id
        # This is the same query as get_package_references, but with an additional filter
        query = f'SELECT {self.columns.reference}, ' \
                f'{self.columns.rrev}, ' \
                f'{self.columns.pkgid}, ' \
                f'{self.columns.prev}, ' \
                f'{self.columns.path}, ' \
                f'MAX({self.columns.timestamp}), ' \
                f'{self.columns.build_id}, ' \
                f'{self.columns.lru} ' \
                f'FROM {self.table_name} ' \
                f"WHERE {self.columns.rrev} = '{ref.revision}' " \
                f"AND {self.columns.reference} = '{str(ref)}' " \
                f"AND {self.columns.build_id} = '{build_id}' " \
                f'GROUP BY {self.columns.pkgid} '

        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()
            if row:
                return self._as_dict(self.row_type(*row))
            return None

    def path_to_ref(self, path):
        query = f'SELECT * FROM {self.table_name} ' \
                f"WHERE {self.columns.path}='{path}'"
        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()
            if not row:
                return None
            ref = RecipeReference.loads(row[0])
            ref.revision = row[1]
            pref = PkgReference(ref, row[2], row[3], row[5])
            return pref
