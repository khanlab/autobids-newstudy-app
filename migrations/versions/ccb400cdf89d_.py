"""empty message

Revision ID: ccb400cdf89d
Revises: bd2e4b1a6955
Create Date: 2021-08-12 09:44:53.338570

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ccb400cdf89d"
down_revision = "bd2e4b1a6955"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tar2bids", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("tar_file_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("bids_file", sa.String(length=200), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_tar2bids_bids_file"), ["bids_file"], unique=False
        )

    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "second_last_pressed_button_id", sa.Integer(), nullable=True
            )
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.drop_column("second_last_pressed_button_id")

    with op.batch_alter_table("tar2bids", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_tar2bids_bids_file"))
        batch_op.drop_column("bids_file")
        batch_op.drop_column("tar_file_id")

    # ### end Alembic commands ###
