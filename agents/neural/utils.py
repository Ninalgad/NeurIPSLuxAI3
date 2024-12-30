import numpy as np
import typing
import torch


class NetworkOutput(typing.NamedTuple):
    value: torch.tensor
    move_policy: torch.tensor
    sap_policy: torch.tensor


def unload(x: torch.tensor):
    return x.detach().cpu().numpy()


def get_distance(pos_a, pos_b):
    # manhattan_distance
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def is_valid_pos(pos):
    return (0 <= pos[0] < 24) and (0 <= pos[1] < 24)


def create_obs_frame(player_obs, player_id, opp_id):
    frames = []

    # map features
    for k in ['energy', 'tile_type']:
        x = player_obs['map_features'][k]
        x = np.expand_dims(x, 0)
        frames.append(x)

    # relic position feature
    relic_frame = np.zeros((1, 24, 24), 'int8')
    for (x, y) in player_obs['relic_nodes']:
        if (x != -1) and (y != -1):
            relic_frame[0, x, y] = 1
    relic_frame -= 1  # added back later
    frames.append(relic_frame)

    # unit energy feature (position is implied)
    unit_frame = np.zeros((1, 24, 24), 'int32')

    for idx in [player_id, opp_id]:
        for (x, y), e in zip(player_obs['units']['position'][idx],
                             player_obs['units']['energy'][idx]):
            if idx == opp_id:
                e = -e
            if (x != -1) and (y != -1):
                unit_frame[0, x, y] = e
    unit_frame -= 1  # added back later
    frames.append(unit_frame)

    # vector information frames
    v_frame = np.zeros((3, 24, 24), 'int32')
    v_frame[0, :, :] = player_obs['match_steps']
    v_frame[1, :, :] = player_obs['team_points'][player_id]
    v_frame[2, :, :] = player_obs['team_points'][player_id] - player_obs['team_points'][opp_id]
    frames.append(v_frame)

    # add 1 to make compression easier
    frames = 1 + np.concatenate(frames, axis=0, dtype='int32')
    return frames
