import os

import torch
from torchvision.models import densenet121, resnet18, resnet34, resnet50

import models.fixcbm as fixcbm
import train.utils as utils


BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
}


def _resolve_backbone(config):
    backbone = config["c_extractor_arch"]
    if not isinstance(backbone, str):
        return backbone
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone!r}.")
    return BACKBONES[backbone]


def construct_model(
        n_concepts,
        n_tasks,
        config,
        c2y_model=None,
        imbalance=None,
        task_class_weights=None,
        intervention_policy=None,
        output_latent=False,
        output_interventions=False,
        **_unused,
):
    if config.get("architecture") != "FixCBM":
        raise ValueError("This release only supports architecture='FixCBM'.")

    backbone = _resolve_backbone(config)
    weight_loss = (
        torch.FloatTensor(imbalance)
        if config.get("weight_loss") and (imbalance is not None)
        else None
    )
    task_weights = (
        torch.FloatTensor(task_class_weights)
        if task_class_weights is not None
        else None
    )

    return fixcbm.FixCBM(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        emb_size=config["emb_size"],
        shared_prob_gen=config.get("shared_prob_gen", True),
        training_intervention_prob=config.get("training_intervention_prob", 0.25),
        embedding_activation=config.get("embedding_activation", "leakyrelu"),
        c2y_model=c2y_model,
        c2y_layers=config.get("c2y_layers", []),
        concept_loss_weight_labeled=config["concept_loss_weight_labeled"],
        concept_loss_weight_unlabeled=config["concept_loss_weight_unlabeled"],
        task_loss_weight=config.get("task_loss_weight", 1.0),
        concept_loss_weight=config.get("concept_loss_weight", 1.0),
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        c_extractor_arch=utils.wrap_pretrained_model(backbone),
        optimizer=config["optimizer"],
        momentum=config.get("momentum", 0.9),
        weight_loss=weight_loss,
        task_class_weights=task_weights,
        top_k_accuracy=config.get("top_k_accuracy"),
        output_latent=output_latent,
        output_interventions=output_interventions,
        intervention_policy=intervention_policy,
        fixmatch_threshold=config.get("fixmatch_threshold", 0.95),
        warmup_epochs=config.get("warmup_epochs", 10),
    )


def load_trained_model(
        config,
        n_tasks,
        result_dir,
        n_concepts,
        imbalance=None,
        task_class_weights=None,
        logger=False,
        accelerator="auto",
        devices="auto",
        intervention_policy=None,
        output_latent=False,
        output_interventions=False,
        enable_checkpointing=False,
        run_name=None,
        **_unused,
):
    del logger, accelerator, devices, enable_checkpointing
    run_name = run_name or config.get("run_name", config["architecture"])
    candidates = [
        os.path.join(result_dir, f"{run_name}.pt"),
        os.path.join(result_dir, f"{config['architecture']}.pt"),
        os.path.join(result_dir, "test.pt"),
    ]
    model_path = next((path for path in candidates if os.path.exists(path)), None)
    if model_path is None:
        tried = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Model file not found. Tried:\n{tried}")

    model = construct_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
        intervention_policy=intervention_policy,
        output_latent=output_latent,
        output_interventions=output_interventions,
    )
    model.load_state_dict(torch.load(model_path))
    return model
