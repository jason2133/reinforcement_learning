import gymnasium as gym

from gymnasium.envs.registration import register

env_dict = gym.envs.registration.registry.copy()
for env in env_dict:
    if "D-8x8-FrozenLake-v0" in env:
        del gym.envs.registration.registry[env]
    elif "S-8x8-FrozenLake-v0" in env:
        del gym.envs.registration.registry[env]


register(
    id="D-8x8-FrozenLake-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": False},
)

register(
    id="S-8x8-FrozenLake-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": True},
)
