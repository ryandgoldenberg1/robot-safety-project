import torch


def get_measure(name):
    return _measures_by_name[name]


def noop(probs, atom_values):
    bs = probs.shape[0]
    return torch.zeros(bs, 1)


def variance(probs, atom_values):
    bs = probs.shape[0]
    num_atoms = atom_values.shape[0]
    assert probs.shape == (bs, num_atoms)
    assert atom_values.shape == (num_atoms,)
    exp_value_sq = (probs * atom_values ** 2).sum(dim=1).unsqueeze(1)
    exp_value = (probs * atom_values).sum(dim=1).unsqueeze(1)
    assert exp_value_sq.shape == (bs, 1)
    assert exp_value.shape == (bs, 1)
    variance = exp_value_sq - exp_value ** 2
    return variance


_measures_by_name = {
    'variance': variance,
    'noop': noop,
}
