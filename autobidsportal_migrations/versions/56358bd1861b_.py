"""empty message

Revision ID: 56358bd1861b
Revises: 5b823749a511
Create Date: 2021-07-26 14:55:37.934932

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "56358bd1861b"
down_revision = "5b823749a511"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("cfmm2tar", schema=None) as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.VARCHAR(length=36),
            type_=sa.Integer(),
            postgresql_using="id::integer",
            existing_nullable=False,
            autoincrement=True,
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("cfmm2tar", schema=None) as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.Integer(),
            type_=sa.VARCHAR(length=36),
            existing_nullable=False,
            autoincrement=True,
        )

    # ### end Alembic commands ###
