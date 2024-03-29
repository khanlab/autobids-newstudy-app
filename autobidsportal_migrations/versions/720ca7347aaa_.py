"""empty message

Revision ID: 720ca7347aaa
Revises: 990a8d4f7c14
Create Date: 2021-08-19 17:39:25.385054

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "720ca7347aaa"
down_revision = "990a8d4f7c14"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("access_to", sa.String(length=128), nullable=True)
        )
        batch_op.drop_column("choices")

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("choices", sa.VARCHAR(length=128), nullable=True)
        )
        batch_op.drop_column("access_to")

    # ### end Alembic commands ###
