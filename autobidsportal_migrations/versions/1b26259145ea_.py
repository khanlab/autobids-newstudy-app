"""empty message

Revision ID: 1b26259145ea
Revises: 56358bd1861b
Create Date: 2021-07-26 15:09:53.779344

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1b26259145ea"
down_revision = "56358bd1861b"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.add_column(sa.Column("success", sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.drop_column("success")

    # ### end Alembic commands ###
