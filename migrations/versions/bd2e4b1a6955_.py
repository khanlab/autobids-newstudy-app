"""empty message

Revision ID: bd2e4b1a6955
Revises: e5b9b2930a6b
Create Date: 2021-08-11 13:23:20.912920

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bd2e4b1a6955'
down_revision = 'e5b9b2930a6b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('tar2bids',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('task_button_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name=op.f('fk_tar2bids_user_id_user')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_tar2bids'))
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('tar2bids')
    # ### end Alembic commands ###