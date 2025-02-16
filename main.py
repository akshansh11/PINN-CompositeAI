import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from pinn_model import CompositePINN

def create_mesh(nx=50, ny=50, nt=1):
    """Create a mesh for visualization"""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    
    X, Y, T = np.meshgrid(x, y, t)
    return X, Y, T

def predict_displacement(model, X, Y, T, phase_fractions):
    """Predict displacement using the trained model"""
    points = torch.tensor(np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1),
                         dtype=torch.float32)
    
    with torch.no_grad():
        displacement = model(points, phase_fractions)
    
    u = displacement[:, 0].reshape(X.shape)
    v = displacement[:, 1].reshape(Y.shape)
    return u, v

def plot_displacement(X, Y, u, v, title):
    """Create displacement plot using plotly"""
    magnitude = np.sqrt(u**2 + v**2)
    
    fig = go.Figure(data=go.Heatmap(
        x=X[:, :, 0],
        y=Y[:, :, 0],
        z=magnitude[:, :, 0],
        colorscale='Viridis',
        colorbar=dict(title='Displacement Magnitude')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='X',
        yaxis_title='Y',
        width=600,
        height=600
    )
    
    return fig

def main():
    st.title("Composite Materials Analysis using Physics-Informed Neural Networks")
    
    # Sidebar for parameters
    st.sidebar.header("Material Parameters")
    
    num_phases = st.sidebar.number_input("Number of Phases", min_value=1, max_value=5, value=2)
    
    phase_fractions = []
    phase_properties = []
    
    for i in range(num_phases):
        st.sidebar.subheader(f"Phase {i+1}")
        fraction = st.sidebar.slider(f"Volume Fraction {i+1}", 0.0, 1.0, 1.0/num_phases)
        E = st.sidebar.number_input(f"Young's Modulus {i+1} (GPa)", value=210.0)
        nu = st.sidebar.number_input(f"Poisson's Ratio {i+1}", value=0.3)
        
        phase_fractions.append(fraction)
        phase_properties.append((E, nu))
    
    # Normalize phase fractions
    phase_fractions = torch.tensor(phase_fractions)
    phase_fractions = phase_fractions / phase_fractions.sum()
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    hidden_layers = st.sidebar.text_input("Hidden Layers", "50,50,50,50")
    hidden_layers = [int(x) for x in hidden_layers.split(",")]
    
    # Create or load model
    if 'model' not in st.session_state:
        model = CompositePINN(hidden_layers=hidden_layers, num_phases=num_phases)
        st.session_state.model = model
    
    # Training button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            X, Y, T = create_mesh()
            points = torch.tensor(np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1),
                                dtype=torch.float32)
            
            # Create a simple dataloader for training
            dataloader = [(torch.tensor(X.flatten()), 
                          torch.tensor(Y.flatten()), 
                          torch.tensor(T.flatten()))]
            
            model = train_model(st.session_state.model, dataloader)
            st.session_state.model = model
            st.success("Model trained successfully!")
    
    # Analysis section
    st.header("Analysis")
    
    # Create mesh for visualization
    X, Y, T = create_mesh()
    
    # Predict displacement
    u, v = predict_displacement(st.session_state.model, X, Y, T, phase_fractions)
    
    # Plot results
    st.subheader("Displacement Field")
    fig = plot_displacement(X, Y, u, v, "Displacement Magnitude")
    st.plotly_chart(fig)
    
    # Display effective properties
    st.subheader("Effective Properties")
    E_eff, nu_eff = st.session_state.model.compute_effective_properties(phase_fractions)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Effective Young's Modulus (GPa)", f"{E_eff.item():.2f}")
    with col2:
        st.metric("Effective Poisson's Ratio", f"{nu_eff.item():.3f}")
    
    # Export results
    if st.button("Export Results"):
        # Create results dictionary
        results = {
            "Effective_Properties": {
                "Young's_Modulus": E_eff.item(),
                "Poisson's_Ratio": nu_eff.item()
            },
            "Phase_Properties": [
                {f"Phase_{i+1}": {
                    "Volume_Fraction": f.item(),
                    "Young's_Modulus": E,
                    "Poisson's_Ratio": nu
                }} for i, (f, (E, nu)) in enumerate(zip(phase_fractions, phase_properties))
            ]
        }
        
        st.download_button(
            "Download Results",
            data=str(results),
            file_name="composite_analysis_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
