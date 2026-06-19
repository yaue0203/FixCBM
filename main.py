import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from pytorch_lightning import seed_everything

from configs.basic_config import get_args
from train.evaluate import evaluate_representation_metrics
from train.training import train_end_to_end_model, update_statistics
from utils import (
    evaluate_expressions,
    generate_dataset_and_update_config,
    generate_hyper_param_configs,
    logging_config,
    zipdir,
)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    with open(f"configs/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    max_epochs = experiment_config.get("max_epochs", 10)
    run_names = [
        run.get("run_name", run.get("architecture", "FixCBM"))
        for run in experiment_config.get("runs", [])
    ]
    run_name_str = "-".join(run_names) if run_names else "FixCBM"

    current_time = time.localtime()
    time_str = time.strftime("%H-%M", current_time)
    date_str = time.strftime("%y%m%d", current_time)
    save_dir = os.path.join(
        args.save_path,
        f"{args.dataset}_{run_name_str}_ep{max_epochs}_{time_str}_{date_str}",
    )
    logging_config(save_dir)

    import logging

    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    logging.info(f"GPU number: {torch.cuda.device_count()}")

    zipf = zipfile.ZipFile(
        file=os.path.join(save_dir, "codes.zip"),
        mode="a",
        compression=zipfile.ZIP_DEFLATED,
    )
    zipdir(Path().absolute(), zipf, include_format=[".py"])
    zipf.close()

    with open(os.path.join(save_dir, "args.yml"), "a") as f:
        yaml.dump(vars(args), f, sort_keys=False)
    with open(os.path.join(save_dir, "experiment_config.yaml"), "w") as f:
        yaml.dump(experiment_config, f)

    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
        task_class_weights,
        acquisition_costs,
    ) = generate_dataset_and_update_config(experiment_config, args)
    del concept_map, intervened_groups, acquisition_costs

    results = defaultdict(dict)
    for current_config in experiment_config["runs"]:
        run_name = current_config.get("run_name", current_config["architecture"])
        trial_config = dict(experiment_config)
        trial_config.update(current_config)

        for run_config in generate_hyper_param_configs(trial_config):
            run_config = dict(run_config)
            run_config["result_dir"] = save_dir
            run_config["c_extractor_arch"] = args.image_encoder
            evaluate_expressions(run_config, soft=True)

            old_results = None
            model, model_results = train_end_to_end_model(
                run_name=run_name,
                task_class_weights=task_class_weights,
                accelerator=args.device,
                devices="auto",
                n_concepts=run_config["n_concepts"],
                n_tasks=run_config["n_tasks"],
                config=run_config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                result_dir=save_dir,
                seed=args.seed,
                imbalance=imbalance,
                old_results=old_results,
                gradient_clip_val=run_config.get("gradient_clip_val", 0),
                activation_freq=args.activation_freq,
                single_frequency_epochs=args.single_frequency_epochs,
            )

            update_statistics(
                aggregate_results=results[run_name],
                run_config=run_config,
                test_results=model_results,
                run_name=run_name,
                prefix="",
            )

            if not run_config.get("skip_repr_evaluation", False):
                update_statistics(
                    aggregate_results=results[run_name],
                    run_config=run_config,
                    model=model,
                    test_results=evaluate_representation_metrics(
                        config=run_config,
                        n_concepts=run_config["n_concepts"],
                        n_tasks=run_config["n_tasks"],
                        test_dl=test_dl,
                        run_name=run_name,
                        imbalance=imbalance,
                        result_dir=save_dir,
                        task_class_weights=task_class_weights,
                        accelerator=args.device,
                        devices="auto",
                        seed=args.seed,
                        old_results=old_results,
                    ),
                    run_name=run_name,
                    prefix="",
                )

            results[run_name]["num_trainable_params"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            results[run_name]["num_non_trainable_params"] = sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            )

        with open(f"{save_dir}/results.txt", "w") as f:
            for key, value in results[run_name].items():
                f.write(f"{key}: {value}\n")

    print("========================finish========================")
