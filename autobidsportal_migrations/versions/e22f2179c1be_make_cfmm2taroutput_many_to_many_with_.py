"""Make Cfmm2tarOutput many-to-many with Tar2bidsOutput

Revision ID: e22f2179c1be
Revises: 4e12d841fcc8
Create Date: 2021-10-14 10:45:17.800952

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e22f2179c1be"
down_revision = "4e12d841fcc8"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "tar2bids_runs",
        sa.Column("cfmm2tar_output_id", sa.Integer(), nullable=False),
        sa.Column("tar2bids_output_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cfmm2tar_output_id"],
            ["cfmm2tar_output.id"],
            name=op.f("fk_tar2bids_runs_cfmm2tar_output_id_cfmm2tar_output"),
        ),
        sa.ForeignKeyConstraint(
            ["tar2bids_output_id"],
            ["tar2bids_output.id"],
            name=op.f("fk_tar2bids_runs_tar2bids_output_id_tar2bids_output"),
        ),
        sa.PrimaryKeyConstraint(
            "cfmm2tar_output_id",
            "tar2bids_output_id",
            name=op.f("pk_tar2bids_runs"),
        ),
    )
    with op.batch_alter_table("tar2bids_output", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_tar2bids_output_cfmm2tar_output_id_cfmm2tar_output",
            type_="foreignkey",
        )
        batch_op.drop_column("cfmm2tar_output_id")

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("tar2bids_output", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("cfmm2tar_output_id", sa.INTEGER(), nullable=False)
        )
        batch_op.create_foreign_key(
            "fk_tar2bids_output_cfmm2tar_output_id_cfmm2tar_output",
            "cfmm2tar_output",
            ["cfmm2tar_output_id"],
            ["id"],
        )

    op.drop_table("tar2bids_runs")
    # ### end Alembic commands ###
