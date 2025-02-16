# pinn_model.py
import torch
import torch.nn as nn
import numpy as np

class ElasticityPINN(nn.Module):
    def __init__(self, hidden_layers=[50, 50, 50, 50]):
        super().__init__()
        
        layers = []
        input_dim = 3  # x, y, t
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.Tanh())
            input_dim = h_dim
            
        layers.append(nn.Linear(input_dim, 2))
        
        self.network = nn.Sequential(*layers)
        self.E = nn.Parameter(torch.tensor(210.0, dtype=torch.float32))
        self.nu = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        
    def forward(self, x):
        return self.network(x)
    
    def compute_strain(self, x, y, t, displacement):
        u = displacement[:, 0:1]
        v = displacement[:, 1:2]
        
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        
        return u_x, v_y, 0.5 * (u_y + v_x)
    
    def compute_stress(self, epsilon_xx, epsilon_yy, epsilon_xy):
        lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        
        sigma_xx = 2 * mu * epsilon_xx + lambda_ * (epsilon_xx + epsilon_yy)
        sigma_yy = 2 * mu * epsilon_yy + lambda_ * (epsilon_xx + epsilon_yy)
        sigma_xy = 2 * mu * epsilon_xy
        
        return sigma_xx, sigma_yy, sigma_xy
    
    def pde_loss(self, x, y, t):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        
        points = torch.stack([x, y, t], dim=1)
        displacement = self.forward(points)
        
        epsilon_xx, epsilon_yy, epsilon_xy = self.compute_strain(x, y, t, displacement)
        sigma_xx, sigma_yy, sigma_xy = self.compute_stress(epsilon_xx, epsilon_yy, epsilon_xy)
        
        momentum_x = torch.autograd.grad(sigma_xx.sum(), x, create_graph=True)[0] + \
                    torch.autograd.grad(sigma_xy.sum(), y, create_graph=True)[0]
        
        momentum_y = torch.autograd.grad(sigma_xy.sum(), x, create_graph=True)[0] + \
                    torch.autograd.grad(sigma_yy.sum(), y, create_graph=True)[0]
        
        return torch.mean(momentum_x**2 + momentum_y**2)

class CompositePINN(ElasticityPINN):
    def __init__(self, hidden_layers=[50, 50, 50, 50], num_phases=2):
        super().__init__(hidden_layers)
        self.num_phases = num_phases
        
        self.E_phases = nn.Parameter(torch.ones(num_phases, dtype=torch.float32) * 210.0)
        self.nu_phases = nn.Parameter(torch.ones(num_phases, dtype=torch.float32) * 0.3)
        
    def update_phase_properties(self, E_values, nu_values):  # THIS IS THE CORRECT METHOD NAME
        """Update phase properties with proper tensor conversion"""
        E_tensor = torch.tensor(E_values, dtype=torch.float32)
        nu_tensor = torch.tensor(nu_values, dtype=torch.float32)
        
        with torch.no_grad():
            self.E_phases.data.copy_(E_tensor)
            self.nu_phases.data.copy_(nu_tensor)
        
    def compute_effective_properties(self, phase_fractions):
        """Compute effective material properties"""
        phase_fractions = torch.tensor(phase_fractions, dtype=torch.float32)
        E_eff = torch.sum(self.E_phases * phase_fractions)
        nu_eff = torch.sum(self.nu_phases * phase_fractions)
        return E_eff, nu_eff
    
    def forward(self, x, phase_fractions=None):
        if phase_fractions is None:
            phase_fractions = torch.ones(self.num_phases, dtype=torch.float32) / self.num_phases
            
        phase_fractions = torch.tensor(phase_fractions, dtype=torch.float32)
        
        with torch.no_grad():
            E_eff, nu_eff = self.compute_effective_properties(phase_fractions)
            self.E.data.copy_(E_eff)
            self.nu.data.copy_(nu_eff)
        
        return super().forward(x)

def train_model(model, dataloader, num_epochs=1000, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            x, y, t = batch
            x = x.float()
            y = y.float()
            t = t.float()
            
            loss = model.pde_loss(x, y, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        losses.append(total_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
    
    return model, losses
