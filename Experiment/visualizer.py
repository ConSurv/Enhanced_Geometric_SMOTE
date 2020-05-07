"""Visualization of re sampling"""

import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize(X, y, X_resampled, y_resampled):
    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)

    # Fit and transform x_resampled to visualise inside a 2D feature space
    X_res_vis = pca.transform(X_resampled)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", color='orange', alpha=0.5, marker='.')
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", color='blue', alpha=0.5, marker='.')
    ax1.set_title('Before ReSampling')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=0.5, marker='.', color='orange')
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=0.5, marker='.', color='blue')
    ax2.set_title('After EG-SMOTE')

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))

    f.legend((c0, c1), ('Class #0', 'Class #1'), loc='lower center', ncol=2)
    plt.tight_layout(pad=3)
    plt.show()
    plt.savefig("output/re_sampled_" + datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S") + ".png", dpi=2000)
