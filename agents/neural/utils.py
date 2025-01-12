import numpy as np
import typing
import torch


class NetworkOutput(typing.NamedTuple):
    value: torch.tensor
    move_policy: torch.tensor
    sap_policy: torch.tensor


def clip_int8(x):
    return np.clip(x, -127, 127)


def unload(x: torch.tensor):
    return x.detach().cpu().numpy()


def get_distance(pos_a, pos_b):
    # manhattan_distance
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def is_valid_pos(pos):
    return (0 <= pos[0] < 24) and (0 <= pos[1] < 24)


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
