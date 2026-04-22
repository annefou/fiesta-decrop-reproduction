rule all:
    input:
        "results/reproduce_decrop_results.json",
        "results/cnn_predictions_val_summary.json",


rule reproduce_test:
    input:
        script="01_reproduce_decrop.py",
    output:
        "results/reproduce_decrop_results.json",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 01_reproduce_decrop.ipynb
        """


rule predict_val:
    input:
        script="02_cnn_val_predictions.py",
    output:
        "results/cnn_predictions_val_summary.json",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 02_cnn_val_predictions.ipynb
        """
