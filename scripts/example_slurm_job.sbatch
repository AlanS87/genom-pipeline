#!/bin/bash --login
#
# Example SLURM job for running a full genom_pipeline task stage-by-stage via
# scripts/genom_stage_runner.py. Copy this file, fill in the placeholders
# below, and adjust the #SBATCH directives and `module load` lines for your
# own cluster -- the partition name, GPU count/type, walltime limit, and
# available modules are all specific to the HPC system you're using (this
# example's values -- a GPU partition, 2 GPUs, a 4-day walltime -- are just a
# starting point, not a requirement of genom_pipeline itself).
#
#SBATCH -p <your-gpu-partition>
#SBATCH -G 2
#SBATCH -t 4-00:00:00
#SBATCH -o logs/genom_%j.log
#SBATCH -e logs/genom_%j.err

# ---- fill these in before submitting ----
CONDA_ENV="/path/to/your/conda/envs/genom"
REPO_DIR="/path/to/genom-pipeline"
TASK_CONFIG="$REPO_DIR/tasks/my_task.yaml"
# Path to a plain-text file containing your OpenAI API key (used by the
# embed stage). Keep this file OUTSIDE the repo and never commit it -- the
# key itself is read into an environment variable below, not hard-coded here.
OPENAI_API_KEY_FILE="/path/to/openai_key.txt"
# ------------------------------------------

# Adjust to whatever your cluster calls these modules (or drop entirely if
# your environment already provides a working conda + JDK 17+ without them).
module load anaconda3 2>/dev/null || true
module load openjdk/17 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "conda activate FAILED for CONDA_ENV='$CONDA_ENV' -- check the path/name is correct and the env actually exists there. Not continuing with the wrong environment."
    exit 1
fi
echo "conda env active: $CONDA_DEFAULT_ENV"
unset PYTHONPATH

export OPENAI_API_KEY="$(cat "$OPENAI_API_KEY_FILE")"

# extract's default JVM heap is 8g (see genom_pipeline/_deeponto_compat.py).
# Raise this if extract fails with a Java heap OutOfMemoryError on a large
# ontology -- consider also setting `reasoner_type: elk` for that ontology
# in the YAML task file (see tasks/example_task.yaml), which is far cheaper
# on memory than the default "hermit" reasoner.
export GENOM_JVM_MEMORY="8g"

cd "$REPO_DIR" || { echo "cannot cd into REPO_DIR=$REPO_DIR"; exit 1; }
mkdir -p logs

echo "== env check =="
which python3
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
java -version

set +e

# Each stage runs as its own OS process (see scripts/genom_stage_runner.py's
# module docstring for why), and every stage is safe to resume: with
# `overwrite: false` in the YAML task file (the default), extract/retrieve/
# judge/fusion skip re-doing work whose output file already exists, and
# define/judge checkpoint their progress incrementally. So if this job runs
# out of wallclock partway through, simply resubmitting this exact same
# script picks up where it stopped -- no changes needed, as long as
# `overwrite: false` stays set for the whole task.
STAGES=(extract define embed retrieve judge fusion)
for stage in "${STAGES[@]}"; do
    echo "== [$stage] =="
    python3 scripts/genom_stage_runner.py "$stage" --config "$TASK_CONFIG"
    rc=$?
    if [ $rc -ne 0 ]; then echo "$stage FAILED rc=$rc"; exit $rc; fi
done

echo "== all stages returned rc=0 =="
