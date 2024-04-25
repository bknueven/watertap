
import os

from pyomo.environ import Objective, Constraint
from pyomo.common.modeling import unique_component_name
from pyomo.repn.plugins.nl_writer import NLWriter

import idaes.core.util.scaling as iscale

def write_nl(model, filename):
    dummy_objective_name = _add_objective_if_needed(model)
    scaling_cache = _cache_scaling_factors(model)

    iscale.constraint_autoscale_large_jac(
        model,
    )

    nl_writer = NLWriter()

    config = nl_writer.config()
    config.symbolic_solver_labels = True
    config.scale_model = False
    config.linear_presolve = False
    
    filename_base = os.path.splitext(filename)[0]
    row_fname = filename_base + '.row'
    col_fname = filename_base + '.col'

    with (open(filename, "w", newline='') as f,
          open(row_fname, "w") as rf,
          open(col_fname, "w") as cf,
         ):
        nl_writer.write(model, f, rf, cf, config=config)

    if dummy_objective_name is not None:
        model.del_component(model.component(dummy_objective_name))
    _reset_scaling_factors(scaling_cache)

def _add_objective_if_needed(model):
    n_obj = 0
    for c in model.component_data_objects(Objective, active=True):
        n_obj += 1
    # Add an objective if there isn't one
    if n_obj == 0:
        _dummy_objective_name = unique_component_name(model, "objective")
        setattr(model, _dummy_objective_name, Objective(expr=0))
        return _dummy_objective_name
    if n_obj > 1:
        raise RuntimeError(f"Multiple objectives are not allowed, found {n_obj} objectives")
    return None

def _cache_scaling_factors(model):
    scaling_cache = [
        (c, iscale.get_scaling_factor(c))
        for c in model.component_data_objects(
            Constraint, active=True, descend_into=True
        )
    ]
    return scaling_cache

def _reset_scaling_factors(scaling_cache):
    for c, s in scaling_cache:
        if s is None:
            iscale.unset_scaling_factor(c)
        else:
            iscale.set_scaling_factor(c, s)
