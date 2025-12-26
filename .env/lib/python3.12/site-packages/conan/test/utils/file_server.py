import os
import uuid

import bottle
from webtest import TestApp

from conan.test.utils.test_files import temp_folder
from conan.internal.util.files import mkdir


class TestFileServer:
    __test__ = False

    def __init__(self, store=None):
        self.store = store or temp_folder(path_with_spaces=False)
        mkdir(self.store)
        self.fake_url = "http://fake%s.com" % str(uuid.uuid4()).replace("-", "")
        self.root_app = bottle.Bottle()
        self.app = TestApp(self.root_app)
        self._attach_to(self.root_app, self.store)

    @staticmethod
    def _attach_to(app, store):
        @app.route("/<file>", method=["GET"])
        def get(file):
            if bottle.request.query:  # For a test using ?q
                file = bottle.request.query["q"]
            mimetype = "application/octet-stream" if file.endswith("tgz") else None
            return bottle.static_file(file, store, mimetype=mimetype)

        @app.route("/<folder>/<file>", method=["PUT"])
        def put(folder, file):
            content = bottle.request.body.read()
            folder = os.path.join(store, folder)
            mkdir(folder)
            with open(os.path.join(folder, file), 'wb') as f:
                f.write(content)

        @app.route("/<folder>/<file>", method=["HEAD"])
        def head(folder, file):
            exists = os.path.exists(os.path.join(store, folder, file))
            if exists:
                return bottle.HTTPResponse(status=200)
            return bottle.HTTPError(404, "Not found")

        @app.route("/forbidden", method=["GET"])
        def get_forbidden():
            return bottle.HTTPError(403, "Access denied.")

        @app.route("/basic-auth/<file>", method=["GET"])
        def get_user_passwd(file):
            auth = bottle.request.auth
            if auth is not None:
                if auth != ("user", "password"):
                    return bottle.HTTPError(401, "Bad credentials")
                return bottle.static_file(file, store)

            auth = bottle.request.headers.get("Authorization")
            if auth is None or auth != "Bearer password":
                return bottle.HTTPError(401, "Not authorized")
            return bottle.static_file(file, store)

        @app.route("/internal_error/<rest>", method=["GET"])
        @app.route("/internal_error", method=["GET"])
        def internal_error():
            return bottle.HTTPError(500, "Internal error")

        @app.route("/gz/<file>", method=["GET"])
        def get_gz(file):
            return bottle.static_file(file, store + "/gz", mimetype="application/octet-stream")

        @app.route("/<folder>/<file>", method=["GET"])
        def get_folder(folder, file):
            return bottle.static_file(file, os.path.join(store, folder))

    def __repr__(self):
        return "TestFileServer @ " + self.fake_url
