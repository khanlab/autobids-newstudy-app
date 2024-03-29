"""empty message

Revision ID: e5b9b2930a6b
Revises: 01c660ea1a21
Create Date: 2021-08-09 13:33:39.307431

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e5b9b2930a6b"
down_revision = "01c660ea1a21"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("cfmm2tar", schema=None) as batch_op:
        batch_op.add_column(sa.Column("date", sa.DateTime(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("cfmm2tar", schema=None) as batch_op:
        batch_op.drop_column("date")

    # ### end Alembic commands ###
