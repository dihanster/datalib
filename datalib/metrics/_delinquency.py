from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.metrics._plot.base import _get_response
from sklearn.utils import (
    check_consistent_length,
    assert_all_finite,
    column_or_1d,
    check_matplotlib_support,
)
from sklearn.base import is_classifier

import numpy as np


class DeliquencyDisplay:
    """Deliquency curve visualization.
    It is recommended to use
    :func:`~datalib.metrics.DeliquencyDisplay.from_estimator` or
    :func:`~datalib.metrics.DeliquencyDisplay.from_predictions`
    to create a `DeliquencyDisplay`. All parameters are stored as attributes.

    Parameters
    ----------
    approval_rate : ndarray of shape (n_bins,)
        The relative percentage population approved.
    default_rate : ndarray of shape (n_bins,)
        The default rate, a.k.a relative percentage of positives on the sample.
    optimal_rate : ndarray of shape (n_bins,)
        The optimal default rate.
    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.
    pos_label : str or int, default=None
        The positive class when computing the calibration curve.
        By default, `estimators.classes_[1]` is considered as the
        positive class.
    Attributes
    ----------
    line_ : matplotlib Artist
        Calibration curve.
    ax_ : matplotlib Axes
        Axes with calibration curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    See Also
    --------
    delinquency_curve : Compute true and predicted probabilities for a
        calibration curve.
    DeliquencyDisplay.from_predictions : Plot calibration curve using true
        and predicted labels.
    DeliquencyDisplay.from_estimator : Plot calibration curve using an
        estimator and data.
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from datalib.metrics import delinquency_curve, DeliquencyDisplay
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> clf = LogisticRegression(random_state=0)
    >>> clf.fit(X_train, y_train)
    LogisticRegression(random_state=0)
    >>> y_prob = clf.predict_proba(X_test)[:, 1]
    >>> approval_rate, default_rate, optimal_rate = delinquency_curve(y_test, y_prob)
    >>> disp = DeliquencyDisplay(approval_rate, default_rate, optimal_rate)
    >>> disp.plot()
    <...>
    """

    def __init__(
        self, approval_rate, default_rate, optimal_rate, *, estimator_name=None, pos_label=None
    ):
        self.approval_rate = approval_rate
        self.default_rate = default_rate
        self.optimal_rate = optimal_rate
        self.estimator_name = estimator_name
        self.pos_label = pos_label

    def plot(self, *, ax=None, name=None, ref_line=True, **kwargs):
        """Plot visualization.
        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.
        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        name : str, default=None
            Name for labeling curve. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.
        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.
        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.
        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("DeliquencyDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name
        info_pos_label = (
            f"(Positive class: {self.pos_label})" if self.pos_label is not None else ""
        )

        line_kwargs = {}
        if name is not None:
            line_kwargs["label"] = name
        line_kwargs.update(**kwargs)

        ref_line_label = "The optimal default rate"
        existing_ref_line = ref_line_label in ax.get_legend_handles_labels()[1]
        if ref_line and not existing_ref_line:
            ax.plot(self.approval_rate, self.optimal_rate, "k:", label=ref_line_label)
        self.line_ = ax.plot(
            self.approval_rate, self.default_rate, "s-", **line_kwargs
        )[0]

        ax.legend(loc="lower right")

        xlabel = f"Relative percentage of approvals on the population {info_pos_label}"
        ylabel = f"Default Rate {info_pos_label}"
        ax.set(xlabel=xlabel, ylabel=ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """Plot deliquency curve using a binary classifier and data.
        A deliquency curve leverages inputs from a binary classifier
        and plots the default rates over unique approval rates, a.k.a
        fractions of the population, on the y-axis.
        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier. The classifier must
            have a :term:`predict_proba` method.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.
        y : array-like of shape (n_samples,)
            Binary target values.
        n_bins : int, default=5
            Number of bins to discretize the [0, 1] interval into when
            calculating the calibration curve. A bigger number requires more
            data.
        pos_label : str or int, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.
            .. versionadded:: 1.1
        name : str, default=None
            Name for labeling curve. If `None`, the name of the estimator is
            used.
        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.
        Returns
        -------
        display : :class:`~datalib.metrics.DeliquencyDisplay`.
            Object that stores computed values.
        See Also
        --------
        DeliquencyDisplay.from_predictions : Plot calibration curve using true
            and predicted labels.
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from datalib.metrics import DeliquencyDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> disp = DeliquencyDisplay.from_estimator(clf, X_test, y_test)
        >>> plt.show()
        """
        method_name = f"{cls.__name__}.from_estimator"
        check_matplotlib_support(method_name)

        if not is_classifier(estimator):
            raise ValueError("'estimator' should be a fitted classifier.")

        y_prob, pos_label = _get_response(
            X, estimator, response_method="predict_proba", pos_label=pos_label
        )

        name = name if name is not None else estimator.__class__.__name__
        return cls.from_predictions(
            y,
            y_prob,
            pos_label=pos_label,
            name=name,
            ref_line=ref_line,
            ax=ax,
            **kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_prob,
        *,
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        """Plot deliquency curve using true labels and predicted probabilities.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.
        y_prob : array-like of shape (n_samples,)
            The predicted probabilities of the positive class.
        pos_label : str or int, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.
            .. versionadded:: 1.1
        name : str, default=None
            Name for labeling curve.
        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.
        Returns
        -------
        display : :class:`~datalib.metrics.DeliquencyDisplay`.
            Object that stores computed values.
        See Also
        --------
        DeliquencyDisplay.from_estimator : Plot calibration curve using an
            estimator and data.
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from datalib.metrics import DeliquencyDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> y_prob = clf.predict_proba(X_test)[:, 1]
        >>> disp = DeliquencyDisplay.from_predictions(y_test, y_prob)
        >>> plt.show()
        """
        method_name = f"{cls.__name__}.from_estimator"
        check_matplotlib_support(method_name)

        approval_rate, default_rate, optimal_rate = delinquency_curve(
            y_true, y_prob, pos_label=pos_label
        )
        name = "Classifier" if name is None else name
        pos_label = _check_pos_label_consistency(pos_label, y_true)

        disp = cls(
            approval_rate=approval_rate,
            default_rate=default_rate,
            optimal_rate=optimal_rate,
            estimator_name=name,
            pos_label=pos_label,
        )
        return disp.plot(ax=ax, **kwargs)


def delinquency_curve(y_true, y_proba, pos_label=None):
    """Delinquency curve for a binary classification.

    The delinquency curve presents the default rate throughout unique approval rates. The `default rate` regards for the percentage
    of actual 1s, a.k.a deliquents, on specified sample; whereas the `approval rate` is the percentage of a sample set as
    approved, or not deliquent.

    Deliquency curve is key on many atuary operations, where grasping the relative percentage of mishaps on approval levels
    is vital.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Correct labels for given dataset.
    y_proba : array, shape = [n_samples]
        Predicted probability scores for the given dataset.
    Returns
    -------
    approval_rate: array.
        An array containing the approval rates used to compute the default_rate
        curve.
    default_rate: array.
        An array containing the default rates values for the approval rates provided in approval_rate.

    """
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba)
    check_consistent_length(y_true, y_proba)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    assert_all_finite(y_true)
    assert_all_finite(y_proba)

    y_true = y_true == pos_label

    scores_idxs = np.argsort(y_proba)[::1]
    actual_idxs = np.argsort(y_true)[::1]
    y_true_sorted_by_scores = y_true[scores_idxs].copy()
    y_true_sorted = y_true[actual_idxs].copy()

    list_index = np.arange(1, len(y_true_sorted_by_scores) + 1)
    approval_rate = np.append(0, list_index / len(list_index))
    default_rate = np.append(0, y_true_sorted_by_scores.cumsum() / list_index)
    optimal_rate = np.append(0, y_true_sorted.cumsum() / list_index)

    return approval_rate, default_rate, optimal_rate
