import sqlite3

from conan.internal.cache.db.table import BaseDbTable
from conan.internal.errors import ConanReferenceDoesNotExistInDB, ConanReferenceAlreadyExistsInDB
from conan.api.model import RecipeReference
from conan.internal.util.dates import timestamp_now


class RecipesDBTable(BaseDbTable):
    table_name = 'recipes'
    columns_description = [('reference', str),
                           ('rrev', str),
                           ('path', str, False, True),
                           ('timestamp', float),
                           ('lru', int)]
    unique_together = ('reference', 'rrev')

    @staticmethod
    def _as_dict(row):
        ref = RecipeReference.loads(row.reference)
        ref.revision = row.rrev
        ref.timestamp = row.timestamp
        return {
            "ref": ref,
            "path": row.path,
            "lru": row.lru,
        }

    def _where_clause(self, ref):
        assert isinstance(ref, RecipeReference)
        where_dict = {
            self.columns.reference: str(ref),
            self.columns.rrev: ref.revision,
        }
        where_expr = ' AND '.join(
            [f"{k}='{v}' " if v is not None else f'{k} IS NULL' for k, v in where_dict.items()])
        return where_expr

    def create(self, path, ref: RecipeReference):
        assert ref is not None
        assert ref.revision is not None
        assert ref.timestamp is not None
        placeholders = ', '.join(['?' for _ in range(len(self.columns))])
        lru = timestamp_now()
        with self.db_connection() as conn:
            try:
                conn.execute(f'INSERT INTO {self.table_name} '
                             f'VALUES ({placeholders})',
                             [str(ref), ref.revision, path, ref.timestamp, lru])
            except sqlite3.IntegrityError:
                raise ConanReferenceAlreadyExistsInDB(f"Reference '{repr(ref)}' already exists")

    def update_timestamp(self, ref: RecipeReference):
        assert ref.revision is not None
        assert ref.timestamp is not None
        query = f"UPDATE {self.table_name} " \
                f"SET {self.columns.timestamp} = '{ref.timestamp}' " \
                f"WHERE {self.columns.reference}='{str(ref)}' " \
                f"AND {self.columns.rrev} = '{ref.revision}' "
        with self.db_connection() as conn:
            conn.execute(query)

    def update_lru(self, ref):
        assert ref.revision is not None
        assert ref.timestamp is not None
        where_clause = self._where_clause(ref)
        lru = timestamp_now()
        query = f"UPDATE {self.table_name} " \
                f"SET {self.columns.lru} = '{lru}' " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            conn.execute(query)

    def remove(self, ref: RecipeReference):
        where_clause = self._where_clause(ref)
        query = f"DELETE FROM {self.table_name} " \
                f"WHERE {where_clause};"
        with self.db_connection() as conn:
            conn.execute(query)

    # returns all different conan references (name/version@user/channel)
    def all_references(self):
        query = f'SELECT DISTINCT {self.columns.reference} FROM {self.table_name}'

        with self.db_connection() as conn:
            r = conn.execute(query)
            rows = r.fetchall()
            return [RecipeReference.loads(row[0]) for row in rows]

    def get_recipe(self, ref: RecipeReference):
        query = f'SELECT * FROM {self.table_name} ' \
                f"WHERE {self.columns.reference}='{str(ref)}' " \
                f"AND {self.columns.rrev} = '{ref.revision}' "
        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()
            if not row:
                raise ConanReferenceDoesNotExistInDB(f"Recipe '{ref.repr_notime()}' not found")
            ret = self._as_dict(self.row_type(*row))
        return ret

    def get_latest_recipe(self, ref: RecipeReference):
        query = f'SELECT {self.columns.reference}, ' \
                f'{self.columns.rrev}, ' \
                f'{self.columns.path}, ' \
                f'MAX({self.columns.timestamp}), ' \
                f'{self.columns.lru} ' \
                f'FROM {self.table_name} ' \
                f"WHERE {self.columns.reference} = '{str(ref)}' " \
                f'GROUP BY {self.columns.reference} '  # OTHERWISE IT FAILS THE MAX()

        with self.db_connection() as conn:
            r = conn.execute(query)
            row = r.fetchone()
            if row is None:
                raise ConanReferenceDoesNotExistInDB(f"Recipe '{ref}' not found")
            ret = self._as_dict(self.row_type(*row))
        return ret

    def get_recipe_revisions_references(self, ref: RecipeReference):
        assert ref.revision is None
        query = f'SELECT * FROM {self.table_name} ' \
                f"WHERE {self.columns.reference} = '{str(ref)}' " \
                f'ORDER BY {self.columns.timestamp} DESC'

        with self.db_connection() as conn:
            r = conn.execute(query)
            ret = [self._as_dict(self.row_type(*row))["ref"] for row in r.fetchall()]
        return ret

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
            ref.timestamp = row[3]
            return ref
