import wandb

# Login if needed
wandb.login()

# Download the artifact
run = wandb.init(project="text-image-aligner-new", entity="FoMo-2025", job_type="download")
artifact = run.use_artifact("FoMo-2025/text-image-aligner-new/model_best.pth:v14", type="model")
artifact_dir = artifact.download()

print(f"Downloaded to: {artifact_dir}")
