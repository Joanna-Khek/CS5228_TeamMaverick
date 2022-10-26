import matplotlib.pyplot as plt
import os 

def highlight_top(ax, orientation, highlight_col):
    if orientation == "horizontal":
        patch_h = [patch.get_width() for patch in ax.patches]
    elif orientation == "vertical":
        patch_h = [patch.get_height() for patch in ax.patches]
    
    max_value = max(patch_h)
    
    # Check if there are more than 1 largest value
    idx_tallest = [i for i, j in enumerate(patch_h) if j == max_value]
        
    for idx in idx_tallest:
        ax.patches[idx].set_facecolor(highlight_col)
        

def save_fig(fig_id, img_dir, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(img_dir, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)