from torch.optim import Adam
from nst.configs.gatys_nst_config import GatysNSTConfig

config = GatysNSTConfig(
    name="elephant-example",
    content_img_path="data/elephant.jpg",
    high_res=True,
    img_dim=(1200, 1200),
    style_img_path="data/african-art.jpg",
    content_layers={15: 1},
    style_layers={3: 0.3, 8: 0.7, 15: 0.7, 22: 0.3},
    epochs=5,
    batches=100,
    optimizer=Adam,
    lr=2.0,
)

config.train()
