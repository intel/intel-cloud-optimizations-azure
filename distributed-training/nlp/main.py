from trainer import Trainer
from omegaconf import DictConfig
from model import GPT
import hydra
import time


def get_time(total_time):
    # Convert total time to hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Format time as hh:mm:ss
    formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return formatted_time


def get_model_and_opt(opt_config):
    print(f"Initializing from OpenAI GPT-2 weights: gpt2")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=0.0)
    model = GPT.from_pretrained("gpt2", override_args)

    optimizer = model.configure_optimizers(
        opt_config.weight_decay,
        opt_config.learning_rate,
        (opt_config.beta1, opt_config.beta2),
        "cpu",
    )
    return model, optimizer


@hydra.main(version_base=None, config_path=".", config_name="intel_nano_gpt_train_cfg")
def main(cfg: DictConfig):
    start = time.time()

    model, optimizer = get_model_and_opt(cfg.optimizer_config)
    trainer = Trainer(
        cfg.trainer_config, cfg.data_dir, model, optimizer, cfg.block_size
    )
    trainer.train()
    end = time.time()

    print(f"Training completed! Total time taken: {get_time(end-start)}")


if __name__ == "__main__":
    main()
