from numpy.typing import NDArray
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from odsmr.sensors import HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech
from odsmr.constants import STATE_BOUNDS

SENSOR_LABELS = [HPC_Tout.__name__, 
                 HP_Nmech.__name__, 
                 HPC_Tin.__name__, 
                 LPT_Tin.__name__, 
                 Fuel_flow.__name__, 
                 HPC_Pout_st.__name__, 
                 LP_Nmech.__name__]

def plot_measures(sequence:NDArray):
    sensor_id = 0
    fig, axs  = plt.subplots(2,4, figsize=(15,8))
    for i in range(2):
        for j in range(4):
            axs[i,j].set_title(SENSOR_LABELS[sensor_id])
            axs[i,j].plot(sequence[:,sensor_id], label=SENSOR_LABELS[sensor_id])
            axs[i,j].grid()
            # axs[i,j].legend()
            sensor_id += 1
            if sensor_id >= len(SENSOR_LABELS):
                break
    plt.legend()
    plt.show()
    
def plot_trajectory(trajectory: NDArray, width=1000, height=800):
    """
    Allows to visualize the generated trajectories for each health indicator (10 in our case)
    
    Parameters
    ----------
    trajectory : NDArray of two dimension (sequence_length, 10)
    """
    indicators = list(STATE_BOUNDS.keys())
    x = list(range(len(trajectory)))

    fig = go.Figure()
    
    for i in range(trajectory.shape[1]):
        fig.add_trace(go.Scatter(
            x=x, y=trajectory[:, i],
            mode='lines+markers',
            name=indicators[i],#f'Indicator {i}',
            #line=dict(color='blue')
        ))

    fig.update_layout(
        title='Degradation Trajectory',
        xaxis_title='Time Step',
        yaxis_title='Indicator Value',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )

    fig.update_layout(
    autosize=False,
    width=width,
    height=height)
    
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01
    ))
    
    fig.show() 