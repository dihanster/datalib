"""
Module containing the implementar for CAP Curve Display.
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.metrics._plot.base import _get_response
from sklearn.utils import check_matplotlib_support

from .. import cap_curve


class CAPCurveDisplay:
    """CAP Curve visualization.

    Parameters
    ----------
    cumulative_gains : ndarray
        Cumulative gain with each threshold (percentage of class 1).

    thresholds : ndarray
        Increasing thresholds (percentage of examples) on the decision
        function used to compute cap curve.

    positive_rate : ndarray
        Rate of positive class examples to compute the perfect curve.

    gini : float, default=None
        Gini score. If None, the gini score is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The class considered as the positive class when computing
        the CAP curve.
        By default, `estimators.classes_[1]` is considered
        as the positive class.

    Attributes
    ----------
    line_ : matplotlib Artist
        CAP Curve.
    ax_ : matplotlib Axes
        Axes with CAP Curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    """

    def __init__(
        self,
        *,
        cumulative_gains,
        thresholds,
        positive_rate=None,
        gini=None,
        estimator_name=None,
        pos_label=None,
    ):
        self.estimator_name = estimator_name
        self.cumulative_gains = cumulative_gains
        self.thresholds = thresholds
        self.positive_rate = positive_rate
        self.gini = gini
        self.pos_label = pos_label

    def plot(
        self,
        *,
        plot_random=False,
        plot_perfect=False,
        name=None,
        ax=None,
        **kwargs
    ):
        """Plot visualization
        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of CAP Curve for labeling. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        plot_random : boolean, default = False
            Flag indicating whether to plot the baseline random curve (True)
            or not (False).

        plot_perfect : boolean, default = False
            Flag indicating whether to plot the baseline perfect curve (True)
            or not (False).

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.CAPCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("CAPCurveDisplay.plot")

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.gini is not None and name is not None:
            line_kwargs["label"] = f"{name} (GINI = {self.gini:0.2f})"
        elif self.gini is not None:
            line_kwargs["label"] = f"Gini = {self.gini:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        if ax is None:
            _, ax = plt.subplots()

        if plot_random is True:
            ax.plot([0, 1], [0, 1], linestyle="--", label="Random Model")

        if plot_perfect is True and self.positive_rate is not None:
            ax.plot(
                [0, self.positive_rate, 1],
                [0, 1, 1],
                label="Perfect Model",
            )

        (self.line_,) = ax.plot(self.thresholds, self.cumulative_gains, **line_kwargs)
        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )

        xlabel = "% of Observations" + info_pos_label
        ylabel = "% of Positive Observations" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

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
        sample_weight=None,
        response_method="auto",
        pos_label=None,
        plot_random=False,
        plot_perfect=False,
        name=None,
        ax=None,
        **kwargs,
    ):
        """Create a CAP Curve display from an estimator.

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        response_method : {'predict_proba', 'decision_function', 'auto'} \
                default='auto'
            Specifies whether to use :term:`predict_proba` or
            :term:`decision_function` as the target response. If set to 'auto',
            :term:`predict_proba` is tried first and if it does not exist
            :term:`decision_function` is tried next.

        pos_label : str or int, default=None
            The class considered as the positive class when computing the roc auc
            metrics. By default, `estimators.classes_[1]` is considered
            as the positive class.

        plot_random : boolean, default = False
            Flag indicating whether to plot the baseline random curve (True)
            or not (False).

        plot_perfect : boolean, default = False
            Flag indicating whether to plot the baseline perfect curve (True)
            or not (False).

        name : str, default=None
            Name of CAP Curve for labeling. If `None`, use the name of the
            estimator.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.CAPCurveDisplay`
            The ROC Curve display.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from datalib.metrics import CAPCurveDisplay
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        >>> clf = SVC(random_state=0).fit(X_train, y_train)
        >>> CAPCurveDisplay.from_estimator(clf, X_test, y_test)
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        name = estimator.__class__.__name__ if name is None else name

        y_score, pos_label = _get_response(
            X,
            estimator,
            response_method=response_method,
            pos_label=pos_label,
        )

        return cls.from_predictions(
            y_true=y,
            y_score=y_score,
            sample_weight=sample_weight,
            pos_label=pos_label,
            plot_random=plot_random,
            plot_perfect=plot_perfect,
            name=name,
            ax=ax,
            **kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_score,
        *,
        sample_weight=None,
        pos_label=None,
        plot_random=False,
        plot_perfect=False,
        name=None,
        ax=None,
        **kwargs,
    ):
        """Plot CAP curve given the true and predicted score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_score : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by “decision_function” on some classifiers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        pos_label : str or int, default=None
            The label of the positive class. When `pos_label=None`, if `y_true`
            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an
            error will be raised.

        plot_random : boolean, default = False
            Flag indicating whether to plot the baseline random curve (True)
            or not (False).

        plot_perfect : boolean, default = False
            Flag indicating whether to plot the baseline perfect curve (True)
            or not (False).

        name : str, default=None
            Name of ROC curve for labeling. If `None`, name will be set to
            `"Classifier"`.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.CAPCurveDisplay`
            Object that stores computed values.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from datalib.metrics import CAPCurveDisplay
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        >>> clf = SVC(random_state=0, probability=True).fit(X_train, y_train)
        >>> y_pred = clf.predict_proba(X_test)[:, 1]
        >>> CAPCurveDisplay.from_predictions(y_test, y_pred)
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_predictions")

        cumulative_gains, thresholds, gini = cap_curve(y_true, y_score, sample_weight)
        positive_rate = np.sum(y_true) / len(y_true) if plot_perfect is True else None

        name = "Classifier" if name is None else name
        pos_label = _check_pos_label_consistency(pos_label, y_true)

        viz = CAPCurveDisplay(
            cumulative_gains=cumulative_gains,
            thresholds=thresholds,
            positive_rate=positive_rate,
            gini=gini,
            estimator_name=name,
            pos_label=pos_label,
        )

        return viz.plot(
            ax=ax,
            name=name,
            plot_random=plot_random,
            plot_perfect=plot_perfect,
            **kwargs,
        )
