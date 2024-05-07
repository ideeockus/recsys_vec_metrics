# recsys_vec_metrics
Dependence free metrics for recommendations, based on user / item vectors

## Calculate all metrics at once
```python
from scipy.spatial import distance

# similarity func can be injected, e.g. scipy.spatial.distance.cosine
metrics_calculator = recsys_vec_metrics.MetricsCalculator(similarity_func=distance.cosine)

# now we can calculate metrics
metrics_calculator.get_all_metrics(
    events_by_attendees.values(),  # actual_interactions - ground truth data
    recomendations_items_by_attendees.values(),  # recommendations_as_items_ids - [items ids that our recsys recommend]
    recomendations_vecs_by_attendees.values(),  # recommendations_as_vecs - [vectors of items that our recsys recommend]
    events_embeddings_by_id,  # embeddings_by_item - vectors by item id
)
```

Output:
```commandline
{
    'Precision@K': 0.0013307330503613628,
    'Recall@K': 0.005500237079184448,
    'F1@k': 0.002142989100314059,
    'Coverage': 0.3037037037037037,
    'Diversity': 6.261786211931749,
    'Personalisation': 0.7450771640582783,
    'Novelty': 0.5695223906605548,
    'Serendipity': 0.001765089763932042
}
```


## Calculate specific metric

### Precision@K
see `calc_precision_k`

### Recall@K
see `calc_recall_k`

### F1@k
see `calc_f1score_k`

### Coverage
see `calc_coverage`

### Diversity
see `calc_diversity`

### Personalisation
see `calc_personalisation`

### Novelty
see `calc_novelty`

### Serendipity
see `calc_serendipity`
