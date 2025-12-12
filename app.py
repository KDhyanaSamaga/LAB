import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Deep Neural Network Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ 
# Define Deep Neural Network 
# ------------------------------ 
class DeepNN(nn.Module): 
    def __init__(self, input_size, hidden1, hidden2, num_classes): 
        super(DeepNN, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU() 
        self.fc3 = nn.Linear(hidden2, num_classes)
 
    def forward(self, x): 
        out = self.fc1(x) 
        out = self.relu1(out) 
        out = self.fc2(out) 
        out = self.relu2(out) 
        out = self.fc3(out) 
        return out

# Title
st.markdown('<div class="main-header">🧠 Deep Neural Network Classifier</div>', unsafe_allow_html=True)
st.markdown("### Iris Dataset Classification with PyTorch")

# Sidebar for hyperparameters
st.sidebar.header("⚙️ Model Configuration")
hidden1 = st.sidebar.slider("Hidden Layer 1 Size", 8, 64, 16, 4)
hidden2 = st.sidebar.slider("Hidden Layer 2 Size", 4, 32, 8, 4)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
epochs = st.sidebar.slider("Number of Epochs", 50, 500, 100, 50)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5) / 100

st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Information")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['Species'] = [iris.target_names[i] for i in y]
    
    st.dataframe(df.head(10), use_container_width=True)
    
    st.info(f"""
    **Dataset Details:**
    - Total Samples: {len(X)}
    - Features: {X.shape[1]}
    - Classes: {len(np.unique(y))} ({', '.join(iris.target_names)})
    """)

with col2:
    st.subheader("🏗️ Network Architecture")
    
    # Display network architecture
    architecture_df = pd.DataFrame({
        'Layer': ['Input', 'Hidden 1', 'ReLU', 'Hidden 2', 'ReLU', 'Output'],
        'Size': [4, hidden1, '-', hidden2, '-', 3],
        'Activation': ['-', 'Linear', 'ReLU', 'Linear', 'ReLU', 'Softmax']
    })
    st.dataframe(architecture_df, use_container_width=True)
    
    # Network visualization
    fig = go.Figure()
    
    layers = [4, hidden1, hidden2, 3]
    layer_names = ['Input\n(4)', f'Hidden 1\n({hidden1})', f'Hidden 2\n({hidden2})', 'Output\n(3)']
    
    for i, (size, name) in enumerate(zip(layers, layer_names)):
        y_positions = np.linspace(0, 1, size)
        x_position = i
        
        fig.add_trace(go.Scatter(
            x=[x_position] * size,
            y=y_positions,
            mode='markers',
            marker=dict(size=20, color=f'rgb({50+i*60}, {100+i*40}, {200-i*40})'),
            name=name,
            showlegend=False
        ))
        
        fig.add_annotation(
            x=x_position,
            y=-0.15,
            text=name,
            showarrow=False,
            font=dict(size=12, color='white')
        )
    
    fig.update_layout(
        title="Neural Network Structure",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Training section
if train_button:
    st.markdown("---")
    st.subheader("🎯 Training Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Initialize model
    model = DeepNN(X.shape[1], hidden1, hidden2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    loss_history = []
    
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    status_text.text("✅ Training Complete!")
    
    # Evaluation
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        train_outputs = model(X_train)
        _, train_predicted = torch.max(train_outputs, 1)
        train_accuracy = (train_predicted == y_train).sum().item() / y_train.size(0)
    
    # Results
    st.markdown("---")
    st.subheader("📈 Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Train Accuracy", f"{train_accuracy*100:.2f}%", 
                 delta=f"{(train_accuracy-0.5)*100:.1f}% vs random")
    
    with col2:
        st.metric("Test Accuracy", f"{accuracy*100:.2f}%",
                 delta=f"{(accuracy-train_accuracy)*100:.1f}% vs train")
    
    with col3:
        st.metric("Final Loss", f"{loss_history[-1]:.4f}",
                 delta=f"-{loss_history[0]-loss_history[-1]:.2f}")
    
    # Loss curve
    col1, col2 = st.columns(2)
    
    with col1:
        fig_loss = px.line(
            x=range(1, epochs+1), 
            y=loss_history,
            labels={'x': 'Epoch', 'y': 'Loss'},
            title='Training Loss Over Time'
        )
        fig_loss.update_traces(line_color='#667eea', line_width=3)
        fig_loss.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test.numpy(), predicted.numpy())
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=iris.target_names,
            y=iris.target_names,
            title='Confusion Matrix',
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Prediction examples
    st.markdown("---")
    st.subheader("🔮 Test Predictions")
    
    results_df = pd.DataFrame({
        'Actual': [iris.target_names[i] for i in y_test.numpy()],
        'Predicted': [iris.target_names[i] for i in predicted.numpy()],
        'Correct': ['✅' if a == p else '❌' for a, p in zip(y_test.numpy(), predicted.numpy())]
    })
    
    st.dataframe(results_df, use_container_width=True)

# Custom prediction section
st.markdown("---")
st.subheader("🧪 Make Custom Predictions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
with col3:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
with col4:
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("🎯 Predict Species", use_container_width=True):
    if 'model' in locals():
        custom_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        custom_input_scaled = scaler.transform(custom_input)
        custom_tensor = torch.tensor(custom_input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(custom_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        st.success(f"### Predicted Species: **{iris.target_names[predicted_class].upper()}**")
        
        prob_df = pd.DataFrame({
            'Species': iris.target_names,
            'Probability': probabilities.numpy() * 100
        })
        
        fig_prob = px.bar(
            prob_df, 
            x='Species', 
            y='Probability',
            title='Prediction Probabilities',
            color='Probability',
            color_continuous_scale='Viridis'
        )
        fig_prob.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("⚠️ Please train the model first using the sidebar button!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with Streamlit 🎈 | PyTorch 🔥 | Plotly 📊</p>
</div>
""", unsafe_allow_html=True)