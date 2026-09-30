# Plant Disease Classification Pipeline

This project classifies PlantVillage leaf images into disease and healthy classes. It is designed as a readable AI/ML internship portfolio project covering data discovery, path-level splitting, TensorFlow preprocessing, transfer learning, focused ablations, lightweight tuning, evaluation, and visual error analysis.

## Dataset

Dataset: [Kaggle PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage). The notebook searches under `/kaggle/input` for a directory containing `train/` and `val/`, merges those folders, and creates its own fixed 70/15/15 train/validation/test split.

The split is performed on file paths before augmentation. The notebook checks filename stems for a defensible repeated-group signal. If no repeated groups are detected, it states the limitation in the output rather than claiming that individual leaves were grouped.

## Models

- A four-block CNN trained from scratch.
- MobileNetV2 with a frozen-backbone stage followed by top-layer fine-tuning.
- EfficientNetB0 with the same two-stage transfer-learning recipe.
- A frozen EfficientNetB0 embedding experiment with a single Dense softmax classifier.

The notebook also includes an extra-contrast preprocessing ablation and a small RandomSearch/handwritten TensorFlow fallback over learning rate, dropout, and unfrozen layers.

## Results

Run `notebooks/plant_disease_pipeline.ipynb` on a Kaggle GPU session before filling this section. All values below are intentionally placeholders and must be replaced with numbers from the generated CSV files.

| Model | Test accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Baseline CNN | `<paste from results/model_comparison.csv>` | `<placeholder>` | `<placeholder>` |
| MobileNetV2 | `<placeholder>` | `<placeholder>` | `<placeholder>` |
| EfficientNetB0 | `<placeholder>` | `<placeholder>` | `<placeholder>` |
| Tuned final model | `<placeholder>` | `<placeholder>` | `<placeholder>` |

Generated artifacts include `results/model_comparison.csv`, `results/per_class_metrics.csv`, `results/confusion_matrix.png`, `results/top_confused_pairs.csv`, `results/high_confidence_wrong_predictions.csv`, `results/gradcam_correct.png`, `results/gradcam_incorrect.png`, and `results/best_model.keras`.

## Error Analysis

After running the notebook, paste the top confused pairs and summarize only patterns supported by the confusion matrix, high-confidence mistakes, and Grad-CAM figures. If the visuals do not support a clear explanation, write `not determined` rather than guessing. Add the generated Grad-CAM images here after the run:

- Correct examples: `results/gradcam_correct.png`
- Incorrect examples: `results/gradcam_incorrect.png`

## Limitations

- PlantVillage images are controlled and may not represent field conditions.
- Filename-based grouping may not identify every near-duplicate leaf; the notebook reports whether its conservative grouping check found evidence.
- A single fixed split and one tuning run do not measure uncertainty across seeds.
- Grad-CAM is an attribution aid, not proof that the model used a biologically meaningful feature.

## Run on Kaggle

1. Create a Kaggle notebook with GPU enabled.
2. Attach the PlantVillage dataset linked above.
3. Upload or clone this project so `notebooks/plant_disease_pipeline.ipynb` is available.
4. Run every cell from top to bottom.
5. Replace the placeholders in this README only with values and figures produced by that run.

## Credits

The downloaded reference notebook informed the broad transfer-learning direction only. Fill these placeholders with the original author's attribution before publishing:

- Original author: `<add name>`
- Original notebook or link: `<add URL>`
