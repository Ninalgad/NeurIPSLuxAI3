import numpy as np
import typing
import torch


class NetworkOutput(typing.NamedTuple):
    value: torch.tensor
    move_policy: torch.tensor
    sap_policy: torch.tensor


def transpose(pos):
    return 23 - pos[1], 23 - pos[0]


def transpose_mat(a):
    return a.copy()[::-1, ::-1].T


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
        m = player_obs['map_features'][k]
        # m = np.maximum(m, transpose_mat(m))  # map features a symmetric
        map_frame[i] = np.clip(m, -128, 126)
    map_frame += 1  # add 1 to make compression easier, since most (hidden) tiles are -1

    # unit energy feature (position is implied)
    unit_frame = np.zeros((1, 24, 24), 'int8')
    for idx in [player_id, opp_id]:
        for (x, y), e in zip(player_obs['units']['position'][idx],
                             player_obs['units']['energy'][idx]):
            if idx == opp_id:
                e = -e
            if (x != -1) and (y != -1):
                unit_frame[0, x, y] = np.clip(e, -127, 127)

    # update hist
    hist_frames = np.roll(hist_frames, 3, axis=0)
    hist_frames[:3] = np.concatenate([unit_frame, map_frame], axis=0, dtype='int8')

    # vector information frames
    v_frames = np.zeros((3, 24, 24), 'int8')
    v_frames[0, :, :] = int(player_obs['match_steps'] / 5)
    v_frames[1, :, :] = np.clip(player_obs['team_points'][player_id], 0, 127)
    v_frames[2, :, :] = np.clip(player_obs['team_points'][player_id] - player_obs['team_points'][opp_id], -127, 127)

    # relic frame
    relic_frame = np.expand_dims(relic_map, 0)

    frames = np.concatenate([relic_frame, v_frames, hist_frames], axis=0, dtype='int8')
    return frames, hist_frames


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    d_model += 1
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, np.newaxis, :, 1:]

    return pos_encoding.astype('float32')
