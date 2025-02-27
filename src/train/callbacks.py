import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 1:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            images, conditions = self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )
            if self.use_wandb:
                samples_list = [wandb.Image(image, caption=f"Sample {i}") for i, (image, _) in enumerate(zip(images, conditions))]
                conditions_list = [wandb.Image(condition, caption=f"Condition {i}") for i, (_, condition) in enumerate(zip(images, conditions))]
                wandb.log({
                    "samples": samples_list,
                    "conditions": conditions_list,
                })

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = trainer.training_config["dataset"]["condition_size"]
        target_size = trainer.training_config["dataset"]["target_size"]
        target_height = int(target_size * trainer.training_config["dataset"]["target_aspect_ratio"])
        condition_height = int(condition_size * trainer.training_config["dataset"]["target_aspect_ratio"])

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        if condition_type == "subject":
            test_list.extend(
                [
                    (
                        Image.open("assets/test_in.jpg"),
                        [0, -32],
                        "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene.",
                    ),
                    (
                        Image.open("assets/test_out.jpg"),
                        [0, -32],
                        "In a bright room. It is placed on a table.",
                    ),
                ]
            )
        elif condition_type == "tryon_old":
            print(condition_type)
            test_list.extend(
                    [
                        (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/train/cloth/dd2046a98255670d6aa2fbad704d1f029132791f.jpg"),
                            [0, -32],
                            "A women over a white background, wearing it",
                        ),
                        (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/test/cloth/2b7e317cdd3efc6cf132364eb1a16cfc04e205a0.jpg"),
                            [0, -32],
                            "A man over a white background. He is wearing it.",
                        ),
                        (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/test/cloth/d1731ae01d13c2e51a30a3f0db5b7d83fd5d2601.jpg"),
                            [0, -32],
                            "Wearing it, a women over a white background.",
                        ),
                        (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/test/cloth/11985de34958ee1ea437cb4b745bd6608c0e8c36.jpg"),
                            [0, -32],
                            "A man on the beach. Wearing it.",
                        ),
                         (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/test/cloth/40da32ac999133aa2bd93c82f93df71a26b72a1c.jpg"),
                            [0, -32],
                            "A man on the beach. Wearing it.",
                        ),
                        (
                            Image.open("/workspace1/pdawson/tryon-scraping/dataset2/test/cloth/0a56420acff8885a8ea5bc6871ff6cab18d3b56f.jpg"),
                            [0, -32],
                            "A man wearing it, over a white background.",
                        )
                    ]
                )
        elif condition_type == "tryon":
            print(condition_type)
            test_list.extend(
                [
                (
                    Image.open("/workspace1/pdawson/tryon-scraping/dataset_zara/train/cloth/13579510850-e2.jpg"),
                    [0, -32],
                    "Wearing it, close-up view of a person's legs in denim jeans, standing against a plain background.",
                ),
                (
                    Image.open("/workspace1/pdawson/tryon-scraping/dataset_zara/train/cloth/13575410800-e1.jpg"),
                    [0, -32],
                    "Wearing it, close-up view of a person's legs in denim jeans, standing against a plain background",
                ),
                (
                    Image.open("/workspace1/pdawson/tryon-scraping/dataset_zara/train/cloth/13511510709-e2.jpg"),
                    [0, -32],
                    "Wearing it, close-up view of a person's legs in denim jeans, standing against a plain background",
                ),
                (
                    Image.open("/workspace1/pdawson/tryon-scraping/dataset_zara/train/cloth/13307510800-e2.jpg"),
                    [0, -32],
                    "Wearing it, close-up view of a person's legs in denim jeans, standing against a plain background",
                ),
                (
                    Image.open("/workspace1/pdawson/tryon-scraping/dataset_zara/train/cloth/13039510800-e2.jpg"),
                    [0, -32],
                    "Wearing it, close-up view of a person's legs in denim jeans, standing against a plain background",
                )
                ]
            )
                        

        elif condition_type == "canny":
            condition_img = Image.open("assets/vase_hq.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = np.array(condition_img)
            condition_img = cv2.Canny(condition_img, 100, 200)
            condition_img = Image.fromarray(condition_img).convert("RGB")
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "coloring":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "depth":
            if not hasattr(self, "deepth_pipe"):
                self.deepth_pipe = pipeline(
                    task="depth-estimation",
                    model="LiheYoung/depth-anything-small-hf",
                    device="cpu",
                )
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "depth_pred":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "deblurring":
            blur_radius = 5
            image = Image.open("./assets/vase_hq.jpg")
            condition_img = (
                image.convert("RGB")
                .resize((condition_size, condition_size))
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "fill":
            condition_img = (
                Image.open("./assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            mask = Image.new("L", condition_img.size, 0)
            draw = ImageDraw.Draw(mask)
            a = condition_img.size[0] // 4
            b = a * 3
            draw.rectangle([a, a, b, b], fill=255)
            condition_img = Image.composite(
                condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "sr":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, -16], "A beautiful vase on a table."))
        elif condition_type == "cartoon":
            condition_img = (
                Image.open("assets/cartoon_boy.png")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, -16], "A cartoon character in a white background. He is looking right, and running."))
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        images = []
        conditions = []

        for i, (condition_img, position_delta, prompt) in enumerate(test_list):
            condition = Condition(
                condition_type=condition_type,
                condition=condition_img.resize(
                    (condition_size, condition_height)
                ).convert("RGB"),
                position_delta=position_delta,
            )
            res = generate(
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_height,
                width=target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
            )
            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            )
            condition_img.save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}_condition.jpg")
            )

            images.append(res.images[0])
            conditions.append(condition_img)
        
        return images, conditions
