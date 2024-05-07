import itertools
import typing

from collections import Counter

# element of recommendation
R_E = typing.TypeVar('R_E', bound=typing.Hashable)

# recommendation
REC = typing.Collection[R_E]

# embedding / vector
E = typing.TypeVar('E', bound=typing.Collection[float])

# similarity function
F_SIM = typing.Callable[[R_E, R_E], float]


class MetricsCalculator:
    def __init__(self, similarity_func: F_SIM):
        self.similarity_func = similarity_func

    def get_all_metrics(
            self,
            actual_interactions: typing.Iterable[REC],
            recommendations_as_items_ids: typing.Iterable[REC],
            recommendations_as_vecs: typing.Iterable[REC],
            embeddings_by_item: typing.Mapping[R_E, E],
    ) -> dict[str, float]:
        return {
            'Precision@K': calc_precision_k(actual_interactions, recommendations_as_items_ids),
            'Recall@K': calc_recall_k(actual_interactions, recommendations_as_items_ids),
            'F1@k': calc_f1score_k(actual_interactions, recommendations_as_items_ids),

            'Coverage': calc_coverage(recommendations_as_items_ids, len(embeddings_by_item)),
            'Diversity': calc_diversity(recommendations_as_vecs, self.similarity_func),
            'Personalisation': calc_personalisation(recommendations_as_items_ids),
            'Novelty': calc_novelty(
                actual_interactions,
                recommendations_as_items_ids,
                embeddings_by_item,
                self.similarity_func,
            ),
            'Serendipity': calc_serendipity(
                actual_interactions,
                recommendations_as_items_ids,
                embeddings_by_item,
                self.similarity_func,
            ),
        }


def calc_precision_k(
        actual_interactions: typing.Iterable[REC],
        recommendations: typing.Iterable[REC],
) -> float:
    """ Precision@K """
    total_recommended_relevant = 0
    total_recommended = 0

    for actual_events, recommended_events in zip(actual_interactions, recommendations):
        relevant_recommendations = set(actual_events) & set(recommended_events)
        total_recommended_relevant += len(relevant_recommendations)
        total_recommended += len(recommended_events)

    precision = total_recommended_relevant / total_recommended
    return precision


def calc_recall_k(
        actual_interactions: typing.Iterable[REC],
        recommendations: typing.Iterable[REC],
) -> float:
    """ Recall@K """

    total_recommended_relevant = 0
    total_relevant = 0

    for actual_events, recommended_events in zip(actual_interactions, recommendations):
        relevant_recommendations = set(actual_events) & set(recommended_events)
        total_recommended_relevant += len(relevant_recommendations)
        total_relevant += len(actual_events)

    recall = total_recommended_relevant / total_relevant
    return recall


def calc_f1score_k(
        actual_interactions: typing.Iterable[REC],
        recommendations: typing.Iterable[REC],
) -> float:
    """ F1Score """
    precision = calc_precision_k(actual_interactions, recommendations)
    recall = calc_recall_k(actual_interactions, recommendations)

    if precision == 0 and recall == 0:
        f1score = 0
    else:
        f1score = 2 * (precision * recall) / (precision + recall)

    return f1score


def calc_coverage(
        recommendations: typing.Iterable[REC],
        total_uniq_items: int,
) -> float:
    """
    Coverage
    is the percent of items that the recommender is able to recommend
    """
    unique_recommendations = set(itertools.chain.from_iterable(recommendations))
    # total_recommended = set(events_by_id.keys())
    # coverage = len(unique_recommendations) / len(total_recommended)
    coverage = len(unique_recommendations) / total_uniq_items
    return coverage


def calc_diversity(
        recommendation_vectors: typing.Iterable[E],
        similarity_func: F_SIM,
) -> float:
    """
    Diversity is inverse proportional to recommendation intra-list similarity.
    Similarity is calculated as average cosine similarity for each-to-each elements
    """
    total_acc = 0
    total_amount = 0

    for rec_vec in recommendation_vectors:
        # recommended_events_vectors = [rec.vector for rec in recommended_events]
        intra_list_sim = calc_intra_list_similarity(rec_vec, similarity_func)
        total_acc += intra_list_sim
        total_amount += 1

    diversity = 1 / (total_acc / total_amount)
    return diversity


def calc_personalisation(
        recommendations: typing.Iterable[REC],
) -> float:
    """
    Personalisation
    calculate as avg frequency of user recommended objects in total recommended objects
    """

    # count of each of recommended object
    recommendations_by_events_counter = Counter(itertools.chain.from_iterable(recommendations))
    total_recommendations = sum(recommendations_by_events_counter.values())

    total_acc = 0
    total_recommendations_len = 0

    for recommended_events in recommendations:
        acc = 0

        for rec_event in recommended_events:
            freq = recommendations_by_events_counter[rec_event] / total_recommendations
            acc += freq

        total_acc += acc
        total_recommendations_len += 1

    personalisation = 1 - (total_acc / total_recommendations_len)
    return personalisation


def calc_novelty(
        actual_interactions: typing.Iterable[REC],
        recommendations: typing.Iterable[REC],
        embeddings_by_item: typing.Mapping[R_E, E],
        similarity_func: F_SIM,
) -> float:
    """
    Novelty / Unexpectedness
    average similarity between two lists items
    """

    acc = 0
    amount = 0

    for actual, recommended in zip(actual_interactions, recommendations):
        actual_embs = [embeddings_by_item[item_id] for item_id in actual]
        recommended_embs = [embeddings_by_item[item_id] for item_id in recommended]

        unexpectedness = calc_lists_similarity(actual_embs, recommended_embs, similarity_func)
        acc += unexpectedness
        amount += 1

    novelty = acc / amount
    return novelty


def calc_serendipity(
        actual_interactions: typing.Iterable[REC],
        recommendations: typing.Iterable[REC],
        embeddings_by_item: typing.Mapping[R_E, E],
        similarity_func: F_SIM,
) -> float:
    """
    Serendipity
    is Unexpectedness * Relevance
    """

    acc = 0
    amount = 0

    for actual_events, recommended_events in zip(actual_interactions, recommendations):
        # calc novelty
        actual_events_embs = [embeddings_by_item[event_id] for event_id in actual_events]
        recommended_events_embs = [embeddings_by_item[event_id] for event_id in recommended_events]

        unexpectedness = calc_lists_similarity(
            actual_events_embs,
            recommended_events_embs,
            similarity_func,
        )

        # calc relevance
        relevance = calc_relevance(actual_events, recommended_events)
        serendipity = unexpectedness * relevance

        acc += serendipity
        amount += 1

    serendipity = acc / amount
    return serendipity


def calc_intra_list_similarity(
        vectors: list[E],
        similarity_func: F_SIM,
) -> float:
    """
    :param vectors:
    :param similarity_func: function to calculate similarity, e.g cosine similarity, euclidean similarity
    :return: calculated intra-list similarity
    """
    acc = 0
    amount = 0

    cur_elem_index = 0
    for i in range(cur_elem_index, len(vectors)):
        cur_elem = vectors[cur_elem_index]
        acc += similarity_func(cur_elem, vectors[i])
        amount += 1

    return acc / amount


def calc_lists_similarity(
        left_list: typing.Iterable[E],
        right_list: typing.Iterable[E],
        similarity_func: F_SIM,
) -> float:
    acc = 0
    amount = 0

    for l_vec in left_list:
        for r_vec in right_list:
            sim = similarity_func(l_vec, r_vec)
            acc += sim
            amount += 1
    return acc / amount


def calc_relevance(
        actual_events: REC,
        recommended_events: REC,
) -> int:
    relevant_recommendations = set(actual_events) & set(recommended_events)
    return len(relevant_recommendations)
