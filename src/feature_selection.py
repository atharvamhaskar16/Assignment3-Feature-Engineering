
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def create_feature_selector(k=8):
    """
    Create a SelectKBest feature selection object
    using Mutual Information.
    """

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )

    return selector
