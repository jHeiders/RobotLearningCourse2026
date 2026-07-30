"""The algorithms a config can select with its ``algo:`` key.

Plain SAC on a task set with a one-hot task id and one shared replay buffer already is
MT-SAC with a shared backbone, so it needs no subclass. To add a variant (disentangled
alphas, PCGrad, a multi-head policy, ...), subclass SB3's ``SAC`` here and add it to the
table; nothing outside this file changes.
"""

from stable_baselines3 import SAC

ALGO_TABLE = {
    "sac": SAC,
}
