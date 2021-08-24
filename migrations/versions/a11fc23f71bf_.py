"""empty message

Revision ID: a11fc23f71bf
Revises: f8442435e6fc
Create Date: 2021-07-20 13:08:36.033145

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a11fc23f71bf"
down_revision = "f8442435e6fc"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.drop_constraint(
            "fk_task_answer_id_answer", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_task_user_id_user"), "user", ["user_id"], ["id"]
        )
        batch_op.drop_column("answer_id")

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("answer_id", sa.INTEGER(), nullable=True)
        )
        batch_op.drop_constraint(
            batch_op.f("fk_task_user_id_user"), type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_task_answer_id_answer", "answer", ["answer_id"], ["id"]
        )
        batch_op.drop_column("user_id")

    # ### end Alembic commands ###
