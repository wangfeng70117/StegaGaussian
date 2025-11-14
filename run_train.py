import os
import subprocess

datasets = {
    "MIP64": ["bicycle", "bonsai", "stump", "counter", "garden", "kitchen", "room"],
    "SYS": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
    "LLFF": ["fern", "flower", "fortress", "horns", "leaves", "llff_room", "orchids", "trex"],
}

extra_args = {
    "SYS": "--white_background"
}

cmd_template = (
    "python train_watermark_lambda.py "
    "-s data/{scene} "
    "-m data/{scene}/output "
    "--water_iterations 40000 "
    "--J 1 "
    "{extra} "
    "--lambda_render 12 "
    "--lambda_wm 12 "
    "--lambda_w 1 "
    "--lambda_neg 15 "
)
fail_dataset = []

for dataset_name, scenes in datasets.items():
    for scene in scenes:
        extra = extra_args.get(dataset_name, "")
        cmd = cmd_template.format(
            scene=scene,
            extra=extra,
            dataset=dataset_name
        )
        print(f"🔧 Executing: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            fail_dataset.append(scene)
            print(f"❌ 命令执行失败: {cmd}")
            print(f"   错误信息: {e}")

print(f'fail_dataset: {fail_dataset}')
