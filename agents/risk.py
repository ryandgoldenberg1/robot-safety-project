import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import agents.utils as utils


def _validate_inputs(atom_values, curr_atom_probs, next_atom_probs, cost, not_done, threshold, curr_return):
    bs = curr_atom_probs.shape[0]
    num_atoms = atom_values.shape[0]
    assert atom_values.shape == (num_atoms,)
    assert curr_atom_probs.shape == (bs, num_atoms)
    assert next_atom_probs.shape == (bs, num_atoms)
    assert cost.shape == (bs, 1)
    assert not_done.shape == (bs, 1)
    assert threshold.shape == (bs, 1)
    assert curr_return.shape == (bs, 1)
    return torch.zeros(bs, 1)


class LinearSchedule:
    def __init__(self, init_value, end_value, num_warmup_steps, num_grow_steps):
        self.init_value = init_value
        self.end_value = end_value
        self.num_warmup_steps = num_warmup_steps
        self.num_grow_steps = num_grow_steps

    def __str__(self):
        return utils.stringify(self)

    def get(self, step):
        if step <= self.num_warmup_steps:
            return self.init_value
        if step > self.num_warmup_steps + self.num_grow_steps:
            return self.end_value
        grow_frac = (step - self.num_warmup_steps) / self.num_grow_steps
        assert 0 <= grow_frac <= 1
        return self.init_value + (self.end_value - self.init_value) * grow_frac


class ZeroRisk(nn.Module):
    def forward(self, atom_values, curr_atom_probs, next_atom_probs, cost, not_done, threshold, curr_return):
        _validate_inputs(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                         cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)
        return torch.zeros(curr_atom_probs.shape[0], 1)


class MeanRisk(nn.Module):
    def __init__(self, discount_factor):
        super().__init__()
        self.discount_factor = discount_factor

    def __str__(self):
        return f"MeanRisk(\n  discount_factor: {self.discount_factor}\n)"

    def forward(self, atom_values, curr_atom_probs, next_atom_probs, cost, not_done, threshold, curr_return):
        _validate_inputs(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                         cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)
        bs = curr_atom_probs.shape[0]
        num_atoms = atom_values.shape[0]
        atom_values = atom_values.expand((bs, num_atoms))
        curr_exp_value = (atom_values * curr_atom_probs).sum(dim=1).unsqueeze(1)
        next_exp_value = (atom_values * next_atom_probs).sum(dim=1).unsqueeze(1)
        assert curr_exp_value.shape == (bs, 1)
        assert next_exp_value.shape == (bs, 1)
        cost_td_error = cost + self.discount_factor * not_done * next_exp_value - curr_exp_value
        return cost_td_error


class ExpUtilityRisk(nn.Module):
    def __init__(self, discount_factor, temperature):
        assert 0 <= discount_factor <= 1
        assert temperature > 0
        super().__init__()
        self.discount_factor = discount_factor
        self.temperature = temperature

    def __str__(self):
        return f"ExpUtilityRisk(\n  discount_factor: {self.discount_factor}\n  temperature: {self.temperature}\n)"

    def _utility_fn(self, atom_values, atom_probs):
        bs, num_atoms = atom_values.shape
        assert atom_values.shape == (bs, num_atoms)
        assert atom_probs.shape == (bs, num_atoms)
        x = torch.exp(self.temperature * atom_values)
        assert x.shape == (bs, num_atoms)
        x = (atom_probs * x).sum(dim=1).unsqueeze(1)
        assert x.shape == (bs, 1)
        x = torch.log(x + 1e-8) / self.temperature
        assert x.shape == (bs, 1)
        return x

    def forward(self, atom_values, curr_atom_probs, next_atom_probs, cost, not_done, threshold, curr_return):
        _validate_inputs(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                         cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)
        bs = curr_atom_probs.shape[0]
        num_atoms = atom_values.shape[0]

        curr_atom_values = atom_values.expand((bs, num_atoms)) - threshold.expand((bs, num_atoms))
        next_atom_values = cost.expand((bs, num_atoms)) + \
            self.discount_factor * not_done * atom_values.expand((bs, num_atoms)) - threshold.expand((bs, num_atoms))
        assert curr_atom_values.shape == (bs, num_atoms)
        assert next_atom_values.shape == (bs, num_atoms)

        curr_utility = self._utility_fn(atom_values=curr_atom_values, atom_probs=curr_atom_probs)
        next_utility = self._utility_fn(atom_values=next_atom_values, atom_probs=next_atom_probs)
        utility_advantage = next_utility - curr_utility
        assert utility_advantage.shape == (bs, 1)
        return utility_advantage


class ClippedVarRisk(nn.Module):
    def __init__(self, discount_factor, margin):
        super().__init__()
        self.discount_factor = discount_factor
        self.margin = margin

    def __str__(self):
        return f"ClippedVarRisk(\n  discount_factor: {self.discount_factor}\n  margin: {self.margin})"

    def _clipped_var(self, atom_values, atom_probs):
        assert atom_values.shape == atom_probs.shape
        assert len(atom_values.shape) == 2
        bs, num_atoms = atom_values.shape
        x = torch.clamp(atom_values, min=0.) ** 2
        assert x.shape == (bs, num_atoms)
        x = (x * atom_probs).sum(dim=1).unsqueeze(1)
        assert x.shape == (bs, 1)
        return x

    def forward(self, atom_values, curr_atom_probs, next_atom_probs, cost, not_done, threshold, curr_return):
        _validate_inputs(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                         cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)
        bs = curr_atom_probs.shape[0]
        num_atoms = atom_values.shape[0]

        curr_atom_values = atom_values.expand((bs, num_atoms)) - threshold.expand((bs, num_atoms)) + self.margin
        next_atom_values = cost.expand((bs, num_atoms)) + \
            self.discount_factor * not_done * atom_values.expand((bs, num_atoms)) - \
            threshold.expand((bs, num_atoms)) + self.margin
        assert curr_atom_values.shape == (bs, num_atoms)
        assert next_atom_values.shape == (bs, num_atoms)

        curr_clipped_var = self._clipped_var(atom_values=curr_atom_values, atom_probs=curr_atom_probs)
        next_clipped_var = self._clipped_var(atom_values=next_atom_values, atom_probs=next_atom_probs)
        clipped_var_advantage = next_clipped_var - curr_clipped_var
        assert clipped_var_advantage.shape == (bs, 1)
        return clipped_var_advantage


def test_clipped_var():
    discount_factor = 1.
    margin = 1.
    var_risk = ClippedVarRisk(discount_factor=discount_factor, margin=margin)

    atom_values = torch.FloatTensor([1., 2., 3., 4.])
    curr_atom_probs = torch.FloatTensor([
        [1., 0., 0., 0.],
        [0.1, 0.5, 0.25, 0.15],
        [0., 0., 0.25, 0.75],
    ])
    next_atom_probs = torch.FloatTensor([
        [0.5, 0.5, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0., 0., 0., 1.],
    ])
    cost = torch.FloatTensor([[1.], [1.], [0.]])
    not_done = torch.FloatTensor([[1.], [0.], [1.]])
    threshold = torch.FloatTensor([[2.], [3.], [4.]])
    curr_return = torch.ones(3, 1)

    res = var_risk(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                   cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)
    exp = torch.FloatTensor([
        [2.5],
        [-0.85],
        [0.25],
    ])

    print('res:', res)
    print('exp:', exp)



def test_linear_schedule():
    constant_schedule = LinearSchedule(init_value=2., end_value=2., num_warmup_steps=0, num_grow_steps=0)
    linear_up = LinearSchedule(init_value=0., end_value=3., num_warmup_steps=10, num_grow_steps=30)
    linear_down = LinearSchedule(init_value=1., end_value=-3., num_warmup_steps=20, num_grow_steps=20)
    no_warmup = LinearSchedule(init_value=-2., end_value=5., num_warmup_steps=0, num_grow_steps=50)

    x = list(range(1, 61))
    plt.plot(x, [constant_schedule.get(i) for i in x], label='constant')
    plt.plot(x, [linear_up.get(i) for i in x], label='linear_up')
    plt.plot(x, [linear_down.get(i) for i in x], label='linear_down')
    plt.plot(x, [no_warmup.get(i) for i in x], label='no_warmup')
    plt.legend()
    plt.show()


def test_mean_risk():
    mean_risk = MeanRisk(discount_factor=1.)
    atom_values = torch.FloatTensor([1., 2., 3., 4.])
    curr_atom_probs = torch.FloatTensor([
        [1., 0., 0., 0.],
        [0.1, 0.5, 0.25, 0.15],
        [0., 0., 0.25, 0.75],
    ])
    next_atom_probs = torch.FloatTensor([
        [0.5, 0.5, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0., 0., 0., 1.],
    ])
    cost = torch.FloatTensor([
        [1.],
        [0.],
        [1.],
    ])
    not_done = torch.FloatTensor([
        [1.],
        [1.],
        [0.],
    ])
    threshold = torch.ones(3, 1) * 2
    curr_return = torch.ones(3, 1)

    res = mean_risk(atom_values=atom_values, curr_atom_probs=curr_atom_probs, next_atom_probs=next_atom_probs,
                    cost=cost, not_done=not_done, threshold=threshold, curr_return=curr_return)

    exp = torch.FloatTensor([
        [-1.5],
        [-0.05],
        [2.75],
    ])
    print('res:', res)
    print('exp:', exp)


def test_utility_risk():
    discount_factor = 1.
    temperature = 2.
    utility = ExpUtilityRisk(discount_factor=discount_factor, temperature=temperature)

    atom_values = torch.FloatTensor([1., 2., 3., 4.])
    curr_atom_probs = torch.FloatTensor([
        [1., 0., 0., 0.],
        [0.1, 0.5, 0.25, 0.15],
        [0., 0., 0.25, 0.75],
    ])
    next_atom_probs = torch.FloatTensor([
        [0.5, 0.5, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0., 0., 0., 1.],
    ])
    cost = torch.FloatTensor([[1.], [1.], [0.]])
    not_done = torch.FloatTensor([[1.], [0.], [1.]])
    threshold = torch.FloatTensor([[2.], [3.], [4.]])
    curr_return = torch.ones(3, 1)

    result = utility(
        atom_values=atom_values,
        curr_atom_probs=curr_atom_probs,
        next_atom_probs=next_atom_probs,
        cost=cost,
        not_done=not_done,
        threshold=threshold,
        curr_return=curr_return
    )

    # exp = torch.FloatTensor([
    #     [log( e^{2* ( 0.5 * 1 )}  ) /2   -   log( e^{ 2*-1 } ) / 2],
    #     [],
    #     [],
    # ])
    exp = torch.FloatTensor([[1.716], [-2.17], [0.12]])

    print('result:', result)
    print('exp:', exp)




def main():
    test_clipped_var()
    # test_utility_risk()
    # test_mean_risk()
    # test_linear_schedule()


if __name__ == '__main__':
    main()
