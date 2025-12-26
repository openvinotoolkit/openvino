import os
import sqlite3
from contextlib import contextmanager
from sqlite3 import OperationalError

from conan.errors import ConanException
from conan.internal.api.remotes import encrypt

REMOTES_USER_TABLE = "users_remotes"
LOCALDB = ".conan.db"

_localdb_encryption_key = os.environ.pop('CONAN_LOGIN_ENCRYPTION_KEY', None)


class LocalDB:

    def __init__(self, dbfolder):
        self.dbfile = os.path.join(dbfolder, LOCALDB)
        self.encryption_key = _localdb_encryption_key

        # Create the database file if it doesn't exist
        if not os.path.exists(self.dbfile):
            par = os.path.dirname(self.dbfile)
            os.makedirs(par, exist_ok=True)
            open(self.dbfile, 'w').close()

            with self._connect() as connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("create table if not exists %s "
                                   "(remote_url TEXT UNIQUE, user TEXT, "
                                   "token TEXT, refresh_token TEXT)" % REMOTES_USER_TABLE)
                except Exception as e:
                    message = f"Could not initialize local sqlite database {self.dbfile}"
                    raise ConanException(message, e)

    def _encode(self, value):
        if value and self.encryption_key:
            return encrypt.encode(value, self.encryption_key)
        return value

    def _decode(self, value):
        if value and self.encryption_key:
            return encrypt.decode(value, self.encryption_key)
        return value

    def clean(self, remote_url=None):
        with self._connect() as connection:
            try:
                cursor = connection.cursor()
                query = "DELETE FROM %s" % REMOTES_USER_TABLE
                if remote_url:
                    query += " WHERE remote_url='{}'".format(remote_url)
                cursor.execute(query)
                try:
                    # https://github.com/ghaering/pysqlite/issues/109
                    connection.isolation_level = None
                    cursor.execute('VACUUM')  # Make sure the DB is cleaned, drop doesn't do that
                except OperationalError:
                    pass
            except Exception as e:
                raise ConanException("Could not initialize local sqlite database", e)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
        connection.text_factory = str
        try:
            yield connection
        finally:
            connection.close()

    def get_login(self, remote_url):
        """ Returns login credentials. This method is also in charge of expiring them. """
        with self._connect() as connection:
            try:
                statement = connection.cursor()
                statement.execute("select user, token, refresh_token from %s where remote_url='%s'"
                                  % (REMOTES_USER_TABLE, remote_url))
                rs = statement.fetchone()
                if not rs:
                    return None, None, None
                name = rs[0]
                token = self._decode(rs[1])
                refresh_token = self._decode(rs[2])
                return name, token, refresh_token
            except Exception:
                raise ConanException("Couldn't read login\n Try removing '%s' file" % self.dbfile)

    def get_username(self, remote_url):
        return self.get_login(remote_url)[0]

    def store(self, user, token, refresh_token, remote_url):
        """ Login is a tuple of (user, token) """
        with self._connect() as connection:
            try:
                token = self._encode(token)
                refresh_token = self._encode(refresh_token)
                statement = connection.cursor()
                statement.execute("INSERT OR REPLACE INTO %s (remote_url, user, token, "
                                  "refresh_token) "
                                  "VALUES (?, ?, ?, ?)" % REMOTES_USER_TABLE,
                                  (remote_url, user, token, refresh_token))
                connection.commit()
            except Exception as e:
                raise ConanException("Could not store credentials %s" % str(e))
