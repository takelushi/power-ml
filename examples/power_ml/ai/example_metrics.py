"""Example of metrics."""

from power_ml.ai import metrics

y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_pred = [1.1, 2.2, 3.8, 4, 5, 2.8, 7.3, 8, 9.3, 10]

# Calculate metric.
result = metrics.MAE(y_true, y_pred)
print(result)


# Custom metric.
# You need to implement calc method.
class CustomMetric(metrics.NumericMetric):
    """Custom metric.

    Match ratio.
    """

    @classmethod
    def calc(cls, y_true, y_pred, **kwargs) -> float:
        """Calculate metric."""
        matchs = [a == b for a, b in zip(y_true, y_pred)]
        return sum(matchs) / len(y_true)


result = CustomMetric(y_true, y_pred)
print(result)
