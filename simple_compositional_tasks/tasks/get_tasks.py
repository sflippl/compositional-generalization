from .addition import Addition
from .context_dependence import ContextDependence
from .transitive_equivalence import TransitiveEquivalence

def get_tasks(cfg):
    return {
        'addition': Addition,
        'context_dependence': ContextDependence,
        'transitive_equivalence': TransitiveEquivalence
    }[cfg.task](cfg=cfg)
