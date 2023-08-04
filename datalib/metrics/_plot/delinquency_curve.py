from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.metrics._plot.base import _get_response
from sklearn.utils import (
    check_matplotlib_support,
)
from sklearn.base import is_classifier
from datalib.metrics import delinquency_curve


class DeliquencyDisplay:
    """Deliquency curve visualization.
    It is recommended to use
    :func:`~datalib.DeliquencyDisplay.from_estimator` or
    :func:`~datalib.DeliquencyDisplay.from_predictions`
    to create a `DeliquencyDisplay`. All parameters are stored as
    attributes.

    Parameters
    ----------
    approval_rate : array-like, shape (n_samples,)
        The relative percentage population approved.

    default_rate : array-like, shape (n_samples,)
        The default rate, a.k.a relative percentage of positives on the
        sample.

    optimal_rate : array-like, shape (n_samples,)
        The optimal default rate.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The positive class when computing the deliquency curve.
        By default, `estimators.classes_[1]` is considered as the
        positive class.

    Attributes
    ----------
    line_ : matplotlib Artist
        Deliquency curve.

    ax_ : matplotlib Axes
        Axes with deliquency curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    delinquency_curve : The main method to calculate needed curves for
    the delinquency analysis.

    DeliquencyDisplay.from_predictions : Plot deliquency curve using
    approval, default, and optimal rates.

    DeliquencyDisplay.from_estimator : Plot deliquency curve using an
    estimator and data.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from datalib import delinquency_curve, DeliquencyDisplay
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> clf = LogisticRegression(random_state=0)
    >>> clf.fit(X_train, y_train)
    LogisticRegression(random_state=0)
    >>> y_prob = clf.predict_proba(X_test)[:, 1]
    >>> approval_rate, default_rate, optimal_rate =
    ...     delinquency_curve(y_test, y_prob)
    >>> disp = DeliquencyDisplay(approval_rate, default_rate, optimal_rate)
    >>> disp.plot()
    <...>
    """

    def __init__(
        self,
        approval_rate,
        default_rate,
        optimal_rate,
        *,
        estimator_name=None,
        pos_label=None,
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
        display : :class:`~datalib.DeliquencyDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("DeliquencyDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name
        info_pos_label = (
            f"(Positive class: {self.pos_label})"
            if self.pos_label is not None
            else ""
        )

        line_kwargs = {}
        if name is not None:
            line_kwargs["label"] = name
        line_kwargs.update(**kwargs)

        ref_line_label = "The optimal Default Rate"
        existing_ref_line = ref_line_label in ax.get_legend_handles_labels()[1]
        if ref_line and not existing_ref_line:
            ax.plot(
                self.approval_rate,
                self.optimal_rate,
                "k:",
                label=ref_line_label,
            )
        self.line_ = ax.plot(
            self.approval_rate, self.default_rate, "s-", **line_kwargs
        )[0]

        ax.legend(loc="lower right")

        xlabel = f"Relative % of approvals on the population {info_pos_label}"
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
            in which the last estimator is a classifier. The classifier
            must have a :term:`predict_proba` method.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Binary target values.

        pos_label : str or int, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.

        name : str, default=None
            Name for labeling curve. If `None`, the name of the estimator
             is used.

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
        display : :class:`~datalib.DeliquencyDisplay`.
            Object that stores computed values.

        See Also
        --------
        DeliquencyDisplay.from_predictions : Plot deliquency curve using
         true and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from datalib import DeliquencyDisplay
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
        """Plot deliquency curve using true labels and predicted
        probabilities.

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
        display : :class:`~datalib.DeliquencyDisplay`.
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
        >>> from datalib import DeliquencyDisplay
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
        return disp.plot(ax=ax, ref_line=ref_line, **kwargs)
