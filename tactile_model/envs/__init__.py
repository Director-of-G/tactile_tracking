from gym.envs.registration import register
from tactile_model.envs.contact_v1 import ContactEnvV1
register(
    id='contact-v1',
    entry_point='tactile_model.envs:ContactEnvV1',
    max_episode_steps=200,
)
