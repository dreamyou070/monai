import os
import shutil
import tempfile
import torch
from monai.config import print_config
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import MeanAbsoluteError, MeanSquaredError, StatsHandler, ValidationHandler, from_engine
from monai.utils import first, set_determinism
from generative.inferers import DiffusionInferer
from generative.engines import DiffusionPrepareBatch
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

print_config()
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

set_determinism(0)
def main() :


    print(f' step 1. make model')
    device = torch.device("cuda")
    model = DiffusionModelUNet(spatial_dims=256,
                               in_channels=3,
                               out_channels=3,
                               num_channels=(64, 128, 128),
                               attention_levels=(False, True, True),
                               num_res_blocks=1,
                               num_head_channels=(0, 128, 128),)
    model.to(device)

    print(f' step 2. scheduler')
    num_train_timesteps = 1000
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)
    n_epochs = 75
    val_interval = 5

    train_handlers = [ValidationHandler(validator=evaluator, interval=val_interval, epoch_level=True),]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=n_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=torch.nn.MSELoss(),
        inferer=inferer,
        prepare_batch=DiffusionPrepareBatch(num_train_timesteps=num_train_timesteps),
        key_train_metric={"train_acc": MeanSquaredError(output_transform=from_engine(["pred", "label"]))},
        train_handlers=train_handlers,)
    trainer.run()

    # %% jupyter={"outputs_hidden": false}
    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    image, intermediates = inferer.sample(input_noise=noise,
                                          diffusion_model=model,
                                          scheduler=scheduler,
                                          save_intermediates=True,
                                          intermediate_steps=100)

    if directory is None:
        shutil.rmtree(root_dir)

if __name__ == '__main__' :
    main()