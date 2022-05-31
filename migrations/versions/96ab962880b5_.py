"""empty message

Revision ID: 96ab962880b5
Revises: 970a0420bce9
Create Date: 2022-05-30 16:15:52.003508

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "96ab962880b5"
down_revision = "970a0420bce9"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "globus_username",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("study_id", sa.Integer(), nullable=False),
        sa.Column("username", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
            name=op.f("fk_globus_username_study_id_study"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_globus_username")),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("globus_username")
    # ### end Alembic commands ###
