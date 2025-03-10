import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from pinn_model import CompositePINN

def create_mesh(nx=50, ny=50, nt=1):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    return np.meshgrid(x, y, t)

def predict_displacement(model, X, Y, T, phase_fractions):
    points = torch.tensor(np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1),
                         dtype=torch.float32)
    
    with torch.no_grad():
        displacement = model(points, phase_fractions)
    
    u = displacement[:, 0].reshape(X.shape)
    v = displacement[:, 1].reshape(Y.shape)
    return u.numpy(), v.numpy()

def plot_displacement(X, Y, u, v, title):
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

def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = CompositePINN(num_phases=2)
        st.session_state.model_initialized = True

def main():
    st.set_page_config(
        page_title="CompositeAI Analyzer",
        page_icon="🔬",
        layout="wide"
    )

    st.markdown("""
        <h1 style='text-align: center; color: #2E4053;'>
            CompositeAI Analyzer
        </h1>
        <h3 style='text-align: center; color: #566573;'>
            Physics-Informed Neural Networks for Multiscale Material Modeling
        </h3>
        <hr>
        """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    st.sidebar.header("Material Parameters")
    
    num_phases = st.sidebar.number_input("Number of Phases", min_value=1, max_value=5, value=2)
    
    phase_fractions = []
    E_values = []
    nu_values = []
    
    for i in range(num_phases):
        st.sidebar.subheader(f"Phase {i+1}")
        fraction = st.sidebar.slider(f"Volume Fraction {i+1}", 0.0, 1.0, 1.0/num_phases)
        E = st.sidebar.number_input(f"Young's Modulus {i+1} (GPa)", value=210.0)
        nu = st.sidebar.number_input(f"Poisson's Ratio {i+1}", value=0.3, min_value=0.0, max_value=0.5)
        
        phase_fractions.append(fraction)
        E_values.append(E)
        nu_values.append(nu)
    
    # Normalize phase fractions
    phase_fractions = np.array(phase_fractions)
    phase_fractions = phase_fractions / phase_fractions.sum()
    
    # Update model if number of phases changed
    if num_phases != st.session_state.model.num_phases:
        st.session_state.model = CompositePINN(num_phases=num_phases)
    
    # Update model properties using the correct method name
    try:
        st.session_state.model.update_phase_properties(E_values, nu_values)  # CORRECT METHOD NAME
    except Exception as e:
        st.error(f"Error updating properties: {str(e)}")
        return
    
    # Training parameters
    st.sidebar.header("Training Parameters")
    num_epochs = st.sidebar.number_input("Number of Epochs", min_value=100, max_value=5000, value=1000, step=100)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                X, Y, T = create_mesh()
                dataloader = [(torch.tensor(X.flatten(), dtype=torch.float32),
                             torch.tensor(Y.flatten(), dtype=torch.float32),
                             torch.tensor(T.flatten(), dtype=torch.float32))]
                
                model, losses = train_model(st.session_state.model, dataloader, 
                                         num_epochs=num_epochs, learning_rate=learning_rate)
                st.session_state.model = model
                
                st.line_chart(losses)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                return
    
    st.header("Analysis")
    
    try:
        X, Y, T = create_mesh()
        u, v = predict_displacement(st.session_state.model, X, Y, T, phase_fractions)
        
        st.subheader("Displacement Field")
        fig = plot_displacement(X, Y, u, v, "Displacement Magnitude")
        st.plotly_chart(fig)
        
        st.subheader("Effective Properties")
        E_eff, nu_eff = st.session_state.model.compute_effective_properties(phase_fractions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Effective Young's Modulus (GPa)", f"{E_eff.item():.2f}")
        with col2:
            st.metric("Effective Poisson's Ratio", f"{nu_eff.item():.3f}")
        
        if st.button("Export Results"):
            results = {
                "Effective_Properties": {
                    "Young's_Modulus": E_eff.item(),
                    "Poisson's_Ratio": nu_eff.item()
                },
                "Phase_Properties": [
                    {f"Phase_{i+1}": {
                        "Volume_Fraction": f,
                        "Young's_Modulus": E,
                        "Poisson's_Ratio": nu
                    }} for i, (f, E, nu) in enumerate(zip(phase_fractions, E_values, nu_values))
                ]
            }
            
            st.download_button(
                "Download Results",
                data=str(results),
                file_name="composite_analysis_results.json",
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
