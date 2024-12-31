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


def create_obs_frame(player_obs, relic_map, hist_frames, player_id, opp_id):

    # map features
    map_frame = np.zeros((2, 24, 24), 'int8')
    for i, k in enumerate(['energy', 'tile_type']):
        map_frame[i] = player_obs['map_features'][k]
    map_frame += 1  # add 1 to make compression easier

    # unit energy feature (position is implied)
    unit_frame = np.zeros((1, 24, 24), 'int32')
    for idx in [player_id, opp_id]:
        for (x, y), e in zip(player_obs['units']['position'][idx],
                             player_obs['units']['energy'][idx]):
            if idx == opp_id:
                e = -e
            if (x != -1) and (y != -1):
                unit_frame[0, x, y] = e

    # update hist
    hist_frame = np.concatenate([unit_frame, map_frame], axis=0, dtype='int32')
    hist_frames = np.roll(hist_frames, 3, axis=0)
    hist_frames[:3] = hist_frame

    # vector information frames
    v_frames = np.zeros((3, 24, 24), 'int32')
    v_frames[0, :, :] = player_obs['match_steps']
    v_frames[1, :, :] = player_obs['team_points'][player_id]
    v_frames[2, :, :] = player_obs['team_points'][player_id] - player_obs['team_points'][opp_id]

    # relic frame
    relic_frame = np.expand_dims(relic_map, 0)

    frames = np.concatenate([relic_frame, v_frames, hist_frames], axis=0, dtype='int32')
    return frames, hist_frames
