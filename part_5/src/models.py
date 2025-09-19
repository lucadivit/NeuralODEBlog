import torch
import torch.nn as nn
from .HarmonicOscillatorData import HarmonicOscillator
from .LotkaVolterraData import LotkaVolterra

def harmonic_oscillator_nets():

    def create_net(is_node: bool):
        input_dim = HarmonicOscillator.get_dim() if is_node else HarmonicOscillator.get_dim() + 1
        return nn.Sequential(
                torch.nn.Linear(input_dim, 16),
                torch.nn.Tanh(),
                torch.nn.Linear(16, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, HarmonicOscillator.get_dim()),
            )

    class SimpleNet(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ann = create_net(is_node=False)
            self.name = HarmonicOscillator.get_name() + "_MLP"

        def forward(self, x, dt):
            z = torch.cat([x, dt], dim=-1)
            return self.ann(z)

    class NODE(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ann = create_net(is_node=True)
            self.method = HarmonicOscillator.get_method()
            self.name = HarmonicOscillator.get_name() + "_NODE"
            self.h = 0.01

        def forward(self, t, state):
            return self.ann(state)

    return SimpleNet(), NODE()

def lotka_volterra_nets():

    def create_net(is_node: bool):
        input_dim = LotkaVolterra.get_dim() if is_node else LotkaVolterra.get_dim() + 1
        return nn.Sequential(
                torch.nn.Linear(input_dim, 16),
                torch.nn.Tanh(),
                torch.nn.Linear(16, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, LotkaVolterra.get_dim()),
            )

    class SimpleNet(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ann = create_net(is_node=False)
            self.name = LotkaVolterra.get_name() + "_MLP"

        def forward(self, x, dt):
            z = torch.cat([x, dt], dim=-1)
            return self.ann(z)

    class NODE(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ann = create_net(is_node=True)
            self.method = LotkaVolterra.get_method()
            self.name = LotkaVolterra.get_name() + "_NODE"
            self.h = 0.01

        def forward(self, t, state):
            return self.ann(state)

    return SimpleNet(), NODE()