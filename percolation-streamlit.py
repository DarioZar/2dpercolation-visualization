import streamlit as st
import numpy as np
from scipy.ndimage import label
import plotly.graph_objects as go

def connected_comp(lat):
    l, _ = label(lat)
    return np.sort(np.unique(l[l!=0], return_counts=True)[1])

@st.cache_data
def simulate(N, ps, iters=100):
    comp = np.zeros([len(ps),3])
    for i in range(iters):
        lat = np.random.rand(N, N)
        components = [connected_comp(lat<p) for p in ps]
        lcc = np.array([c[-1] if len(c)>0 else 0 for c in components])
        slcc = np.array([c[-2] if len(c)>1 else 0 for c in components])
        mean = np.array([np.mean(c) for c in components])
        comp += np.vstack([np.array([lcc, slcc, mean]).T])
    return comp/iters

    
st.title("Percolation 2D Lattice")

st.write("This is a simple simulation of percolation in a 2D lattice.\
         The lattice is a square grid of size N x N. Each site in the lattice is either occupied with probability p\
         or empty with probability 1-p. The goal is to determine the probability that the system percolates, i.e.,\
         that there is a path of occupied sites from the top to the bottom of the lattice.")

with st.sidebar:
    N = st.number_input("Size of the lattice", min_value=10, value=100, max_value=1000, step=10)
    p = st.slider("Occupation probability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    Nps = st.number_input("Number of probability points", min_value=10, value=100, max_value=1000, step=10)
    Nr = st.number_input("Number of realizations", min_value=1, value=1, max_value=100, step=10)


lattice = np.random.rand(N, N)

st.write("## Occupation matrix")
heatmap = go.Figure(data=go.Heatmap(z=(lattice < p).astype(int), colorscale='gray'))
heatmap.update_layout(title='Occupation Matrix', xaxis_title='X', yaxis_title='Y',
                      autosize=True, height=700)
heatmap.update_traces(showscale=False)
st.plotly_chart(heatmap, use_container_width=True)

st.write("Dimensions of the connected components")
components = connected_comp(lattice<p)
st.write("Largest component:", components[-1]/N**2 * 100, "%")
st.write("Second largest component:", components[-2]/N**2 * 100, "%")
st.write("Mean component size:", np.mean(components)/N**2 * 100, "%")

st.write("## Show Connected Components at different occupation probabilities:")

ps = np.linspace(0.01, 1, int(Nps))
components = simulate(N, ps, Nr)

lcc = components[:,0]
slcc = components[:,1]
mean = components[:,2]

fig_lcc = go.Figure(data=go.Scatter(x=ps, y=lcc, mode='lines', name='Largest Component'))
fig_lcc.update_layout(title='Largest Connected Component', xaxis_title='Occupation Probability', yaxis_title='Size')
fig_lcc.add_vline(x=0.5927, line_dash="dash", line_color="red", annotation_text="Percolation: 0.5927", annotation_position="top left")

fig_slcc = go.Figure(data=go.Scatter(x=ps, y=slcc, mode='lines', name='Second Largest Component'))
fig_slcc.update_layout(title='Second Largest Connected Component', xaxis_title='Occupation Probability', yaxis_title='Size')
fig_slcc.add_vline(x=0.5927, line_dash="dash", line_color="red", annotation_text="Percolation: 0.5927", annotation_position="top left")

fig_mean = go.Figure(data=go.Scatter(x=ps, y=mean, mode='lines', name='Mean Component Size'))
fig_mean.update_layout(title='Mean Component Size', xaxis_title='Occupation Probability', yaxis_title='Size')

st.plotly_chart(fig_lcc, use_container_width=True)
st.plotly_chart(fig_slcc, use_container_width=True)
st.plotly_chart(fig_mean, use_container_width=True)
