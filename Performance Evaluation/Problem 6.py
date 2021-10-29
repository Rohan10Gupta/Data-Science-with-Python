import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd


def liftChart(predicted, title='Decile Lift Chart', labelBars=True, ax=None, figsize=None):
    groups = [int(10 * i / len(predicted)) for i in
              range(len(predicted))]
    meanPercentile = predicted.groupby(groups).mean()
    meanResponse = meanPercentile / predicted.mean()
    meanResponse.index = (meanResponse.index + 1) * 10
    ax = meanResponse.plot.bar(color='C0', ax=ax,
                               figsize=figsize)
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)
    if labelBars:
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()),
                        (p.get_x(), p.get_height() + 0.1))
    return ax


if __name__ == '__main__':
    y_test = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    pred = pd.Series([0.03, 0.52, 0.38, 0.82, 0.33, 0.42, 0.55, 0.59, 0.09, 0.21, 0.43, 0.04, 0.08, 0.13, 0.01, 0.079, 0.42, 0.29, 0.08, 0.02])
    pred = pred.sort_values(ascending=False)
    liftChart(pred)
    plt.show()
