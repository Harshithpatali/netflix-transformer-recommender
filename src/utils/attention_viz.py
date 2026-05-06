import seaborn as sns
import matplotlib.pyplot as plt

def plot_attention(attn):
    attn = attn[0].mean(0).detach().cpu().numpy()

    fig, ax = plt.subplots()
    sns.heatmap(attn, ax=ax)
    ax.set_title("Attention Heatmap")

    return fig