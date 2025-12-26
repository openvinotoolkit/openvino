import sqlite3
import threading
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from typing import Tuple, List


class BaseDbTable:
    table_name: str = None
    columns_description: List[Tuple[str, type]] = None
    row_type: namedtuple = None
    columns: namedtuple = None
    unique_together: tuple = None
    _lock: threading.Lock = None
    _lock_storage = defaultdict(threading.Lock)

    def __init__(self, filename):
        self.filename = filename
        column_names: List[str] = [it[0] for it in self.columns_description]
        self.row_type = namedtuple('_', column_names)
        self.columns = self.row_type(*column_names)
        self._lock = self._lock_storage[self.filename]

    @contextmanager
    def db_connection(self):
        assert self._lock.acquire(timeout=20), "Conan failed to acquire database lock"
        connection = sqlite3.connect(self.filename, isolation_level=None, timeout=20)
        try:
            yield connection
        finally:
            connection.close()
            self._lock.release()

    def create_table(self):
        def field(name, typename, nullable=False, unique=False):
            field_str = name
            if typename is str:
                field_str += ' text'
            elif typename is int:
                field_str += ' integer'
            else:
                assert typename is float, f"sqlite3 type not mapped for type '{typename}'"
                field_str += ' real'

            if not nullable:
                field_str += ' NOT NULL'

            if unique:
                field_str += ' UNIQUE'

            return field_str

        fields = ', '.join([field(*it) for it in self.columns_description])
        guard = 'IF NOT EXISTS'
        table_checks = f", UNIQUE({', '.join(self.unique_together)})" if self.unique_together else ''
        with self.db_connection() as conn:
            conn.execute(f"CREATE TABLE {guard} {self.table_name} ({fields} {table_checks});")
