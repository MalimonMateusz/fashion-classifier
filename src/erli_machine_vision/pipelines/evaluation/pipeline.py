"""Pipeline 'evaluation': metryki kazdego modelu, porownanie, raport dla najlepszego."""

from functools import partial, update_wrapper

from kedro.pipeline import Node, Pipeline

from erli_machine_vision.model_zoo import MODELS

from .nodes import compare_models, evaluate_model, report_best_model


def _evaluator(model_name: str):
    func = partial(evaluate_model, model_name=model_name)
    return update_wrapper(func, evaluate_model)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        Node(
            _evaluator(name),
            inputs=[f"{name}_model", f"{features}_features", "model_input_table"],
            outputs=[f"{name}_metrics", f"{name}_predictions"],
            name=f"evaluate_{name}",
        )
        for name, features in MODELS.items()
    ]
    nodes += [
        Node(
            compare_models,
            inputs=[f"{name}_metrics" for name in MODELS],
            outputs=["model_comparison", "model_comparison_plot", "best_model_info"],
            name="compare_models",
        ),
        Node(
            report_best_model,
            inputs=[
                "best_model_info",
                "model_input_table",
                "params:evaluation",
                *[f"{name}_predictions" for name in MODELS],
            ],
            outputs=[
                "confusion_matrix_plot",
                "top_errors_plot",
                "classification_report",
            ],
            name="report_best_model",
        ),
    ]
    return Pipeline(nodes)
