rule all:
    input:
        "results/reproduce_decrop_results.json",


rule run_notebook:
    input:
        script="01_reproduce_decrop.py",
    output:
        "results/reproduce_decrop_results.json",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 01_reproduce_decrop.ipynb
        """
