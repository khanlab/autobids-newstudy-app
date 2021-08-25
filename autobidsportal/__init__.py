"""Initialize flask and all its plugins"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_mail import Mail
import flask_excel as excel
from redis import Redis
import rq

app = Flask(__name__)
excel.init_excel(app)
app.config.from_object(os.environ["AUTOBIDSPORTAL_CONFIG"])
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=convention)
app.redis = Redis.from_url(app.config["REDIS_URL"])
app.task_queue = rq.Queue(connection=app.redis)
db = SQLAlchemy(app, metadata=metadata)
migrate = Migrate(app, db, render_as_batch=True, compare_type=True)
login = LoginManager(app)
login.login_view = "login"
mail = Mail(app)
bootstrap = Bootstrap(app)

# Will be fixed by an application factory
from autobidsportal import routes, models, errors  # pylint: disable=C0413
