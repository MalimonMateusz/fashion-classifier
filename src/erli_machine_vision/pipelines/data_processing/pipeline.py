"""Pipeline 'data_processing': surowe dane ze scrapera -> czysty, podzielony zbior."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    dedupe_and_group,
    plot_class_distribution,
    plot_samples,
    split_dataset,
    validate_images,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                validate_images,
                inputs=["raw_products", "params:data_processing.validation"],
                outputs="validated_products",
                name="validate_images",
            ),
            Node(
                dedupe_and_group,
                inputs=["validated_products", "params:data_processing.dedup"],
                outputs=["clean_products", "dedup_stats"],
                name="dedupe_and_group",
            ),
            Node(
                split_dataset,
                inputs=["clean_products", "params:data_processing.split"],
                outputs="model_input_table",
                name="split_dataset",
            ),
            Node(
                plot_class_distribution,
                inputs="model_input_table",
                outputs="class_distribution_plot",
                name="plot_class_distribution",
            ),
            Node(
                plot_samples,
                inputs=["model_input_table", "params:data_processing.report"],
                outputs="samples_plot",
                name="plot_samples",
            ),
        ]
    )
